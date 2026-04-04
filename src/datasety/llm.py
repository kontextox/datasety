"""Shared LLM backend abstraction for datasety commands."""

import base64
import io
import json
import os
import re
import sys


def resolve_llm_api_config(model_override=None):
    """Resolve OpenAI-compatible API configuration from args and env vars.

    Returns:
        (api_key, base_url, model) tuple.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = (
        os.environ.get("OPENAI_BASE_URL", "")
        or os.environ.get("OPENAI_API_BASE", "")
        or "https://api.openai.com/v1"
    )
    model = model_override or os.environ.get("OPENAI_MODEL", "") or "gpt-5-nano"
    return api_key, base_url, model


def _pil_to_data_url(image):
    """Convert a PIL Image to a base64 data URL."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _model_output_modalities(model: str) -> list:
    """Return the correct output modalities list for a given model.

    OpenRouter docs:
    - Models that output both text and images (Gemini): ["image", "text"]
    - Models that only output images (Flux, Sourceful, etc.): ["image"]
    """
    name = model.lower()
    if "gemini" in name or "google/" in name:
        return ["image", "text"]
    return ["image"]


def _dims_to_aspect_ratio(width: int, height: int) -> str:
    """Convert pixel dimensions to the nearest OpenRouter aspect-ratio string.

    Supported: 1:1, 2:3, 3:2, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9
    """
    candidates = {
        "1:1": (1, 1),
        "2:3": (2, 3),
        "3:2": (3, 2),
        "3:4": (3, 4),
        "4:3": (4, 3),
        "4:5": (4, 5),
        "5:4": (5, 4),
        "9:16": (9, 16),
        "16:9": (16, 9),
        "21:9": (21, 9),
    }
    target = width / height
    best = min(candidates.items(), key=lambda kv: abs(kv[1][0] / kv[1][1] - target))
    return best[0]


def _generate_image_via_api(
    prompt,
    api_key,
    base_url,
    model,
    input_image=None,
    seed=None,
    width=None,
    height=None,
    aspect_ratio=None,
    image_size=None,
):
    """Generate an image via OpenAI-compatible API (OpenRouter, etc).

    For text-to-image: just prompt.
    For image-to-image: prompt + input_image (PIL Image).

    aspect_ratio: OpenRouter aspect-ratio string e.g. "16:9" (overrides width/height).
    image_size: OpenRouter size string e.g. "1K", "2K", "4K".
    width/height: pixel dimensions — converted to nearest aspect_ratio if no
                  explicit aspect_ratio given.
    Returns PIL Image.
    """
    import urllib.request

    from PIL import Image

    url = f"{base_url.rstrip('/')}/chat/completions"

    # Build message content
    content = []
    if input_image is not None:
        data_url = _pil_to_data_url(input_image)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": data_url},
            }
        )
    content.append({"type": "text", "text": prompt})

    # Determine modalities based on model
    modalities = _model_output_modalities(model)

    payload = {
        "model": model,
        "modalities": modalities,
        "messages": [
            {"role": "user", "content": content},
        ],
    }

    if seed is not None:
        payload["seed"] = seed

    # Build image_config: aspect_ratio + optional image_size
    image_config = {}
    ar = aspect_ratio
    if ar is None and width and height:
        ar = _dims_to_aspect_ratio(width, height)
    if ar:
        image_config["aspect_ratio"] = ar
    if image_size:
        image_config["image_size"] = image_size
    if image_config:
        payload["image_config"] = image_config

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())

    # Try structured image response first:
    # choices[0].message.images[0].image_url.url
    msg = data["choices"][0]["message"]
    images = msg.get("images") or []
    if images:
        img_url = images[0].get("image_url", {}).get("url", "")
        if img_url:
            b64_data = img_url.split(",", 1)[-1]
            return Image.open(io.BytesIO(base64.b64decode(b64_data)))

    # Fallback: extract base64 data URL from markdown in content field
    text_content = msg.get("content", "") or ""
    if isinstance(text_content, list):
        # content may be a list of blocks
        for block in text_content:
            if isinstance(block, dict):
                inner = block.get("image_url", {}).get("url", "")
                if inner.startswith("data:image"):
                    b64_data = inner.split(",", 1)[-1]
                    return Image.open(io.BytesIO(base64.b64decode(b64_data)))
        text_content = " ".join(b.get("text", "") for b in text_content if isinstance(b, dict))
    m = re.search(r"data:image/[^;]+;base64,([A-Za-z0-9+/=]+)", text_content)
    if m:
        return Image.open(io.BytesIO(base64.b64decode(m.group(1))))

    raise RuntimeError(f"No image found in API response: {json.dumps(data)[:500]}")


class _OpenAIBackend:
    """OpenAI-compatible API backend."""

    def __init__(self, base_url=None, model=None, api_key=None):
        resolved_key, resolved_url, resolved_model = resolve_llm_api_config(model)
        self.base_url = base_url or resolved_url
        self.model = model or resolved_model
        self.api_key = api_key or resolved_key
        if not self.api_key:
            print("Error: OPENAI_API_KEY environment variable is required for --llm-api")
            sys.exit(1)

    def generate(self, system_prompt, user_prompt):
        import urllib.request

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.9,
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


class _OllamaBackend:
    """Local Ollama server backend."""

    def __init__(self, model, base_url=None):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_HOST", "") or "http://localhost:11434"

    def generate(self, system_prompt, user_prompt):
        import urllib.request

        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            }
        ).encode()

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        return data["message"]["content"]


class _GGUFBackend:
    """Local GGUF model via llama-cpp-python."""

    def __init__(self, model_path):
        self.model_path = model_path
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        try:
            from llama_cpp import Llama
        except ImportError:
            print("Error: llama-cpp-python is required for --llm-gguf")
            print("Run: pip install llama-cpp-python")
            sys.exit(1)

        from datasety.common import _resolve_hf_file

        resolved = _resolve_hf_file(self.model_path)
        print(f"Loading GGUF model: {resolved}")
        self._llm = Llama(model_path=resolved, n_ctx=4096, verbose=False)

    def generate(self, system_prompt, user_prompt):
        self._load()
        output = self._llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.9,
        )
        return output["choices"][0]["message"]["content"]


class _HFModelBackend:
    """HuggingFace transformers model backend."""

    def __init__(self, model_name):
        self.model_name = model_name
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
        except ImportError:
            print("Error: transformers is required for --llm-model")
            print("Run: pip install transformers")
            sys.exit(1)

        print(f"Loading HF model: {self.model_name}")
        self._pipeline = pipeline(
            "text-generation",
            model=self.model_name,
            device_map="auto",
        )

    def generate(self, system_prompt, user_prompt):
        self._load()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # Attempt to disable thinking mode for Qwen3/Qwen3.5 models.
        # `tokenizer_kwargs` is passed to apply_chat_template; silently ignored
        # for models whose tokenizer doesn't support `enable_thinking`.
        try:
            output = self._pipeline(
                messages,
                max_new_tokens=2048,
                temperature=0.9,
                do_sample=True,
                tokenizer_kwargs={"enable_thinking": False},
            )
        except (TypeError, ValueError):
            output = self._pipeline(
                messages,
                max_new_tokens=2048,
                temperature=0.9,
                do_sample=True,
            )
        text = output[0]["generated_text"][-1]["content"]
        # Strip any remaining Qwen3/Qwen3.5-style thinking blocks (<think>…</think>)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        # Filter out meta-commentary lines (reasoning exposed as plain text by thinking models)
        _reasoning_prefixes = (
            "wait,",
            "let me",
            "let's",
            "i will",
            "i'll",
            "i need",
            "actually,",
            "looking",
            "final check",
            "ok,",
            "okay,",
            "now,",
            "so,",
            "note:",
            "note that",
            "important:",
            "step ",
            "thinking",
            "reasoning",
            "analysis",
            "conclusion",
            "let me check",
            "i should",
            "here are",
            "here is",
            "the prompts",
            "these prompts",
            "revised",
            "draft",
            "constraint",
            "output one",
            "nothing else",
            "character:",
            "format:",
            "variation:",
            "style:",
            "task:",
            "instruction:",
            "prompt:",
            "goal:",
            "requirement:",
            "idea ",
            "option ",
            "version ",
            "example ",
        )
        filtered_lines = []
        seen = set()
        for line in text.splitlines():
            # Strip leading bullet chars (*、•、-) and whitespace
            stripped = re.sub(r"^[\s\*\-•]+", "", line).strip()
            # Strip leading "N: " or "N. " or "N) " numbering prefix from bullet prompts
            stripped = re.sub(r"^\d+[.:)]\s*", "", stripped).strip()
            low = stripped.lower()
            if not stripped:
                continue
            # Skip lines that start with reasoning words
            if any(low.startswith(p) for p in _reasoning_prefixes):
                continue
            # Skip very short lines (likely meta-text, not prompts)
            if len(stripped) < 30:
                continue
            # Deduplicate
            if stripped not in seen:
                seen.add(stripped)
                filtered_lines.append(stripped)
        return "\n".join(filtered_lines) if filtered_lines else text


def _create_llm_backend(args):
    """Create the appropriate LLM backend from CLI args."""
    if args.llm_api:
        return _OpenAIBackend()
    elif args.llm_ollama:
        return _OllamaBackend(args.llm_ollama)
    elif args.llm_gguf:
        return _GGUFBackend(args.llm_gguf)
    elif args.llm_model:
        return _HFModelBackend(args.llm_model)
    return None


def add_llm_arguments(parser):
    """Add standard LLM backend arguments to an argparse parser.

    Adds a mutually exclusive group with --llm-api, --llm-ollama,
    --llm-gguf, and --llm-model.
    """
    llm_group = parser.add_mutually_exclusive_group()
    llm_group.add_argument(
        "--llm-api",
        action="store_true",
        help="Use OpenAI-compatible API (needs OPENAI_API_KEY env var)",
    )
    llm_group.add_argument(
        "--llm-ollama",
        default="",
        metavar="MODEL",
        help="Use local Ollama server with specified model",
    )
    llm_group.add_argument(
        "--llm-gguf", default="", metavar="PATH", help="Use local GGUF model file"
    )
    llm_group.add_argument(
        "--llm-model",
        default="",
        metavar="REPO",
        help="Use HuggingFace model for prompt generation",
    )
