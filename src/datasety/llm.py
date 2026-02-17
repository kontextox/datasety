"""Shared LLM backend abstraction for datasety commands."""

import json
import os
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
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.9,
        }).encode()

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
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }).encode()

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
        output = self._pipeline(
            messages,
            max_new_tokens=2048,
            temperature=0.9,
            do_sample=True,
        )
        return output[0]["generated_text"][-1]["content"]


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
        help="Use OpenAI-compatible API (needs OPENAI_API_KEY env var)"
    )
    llm_group.add_argument(
        "--llm-ollama",
        default="",
        metavar="MODEL",
        help="Use local Ollama server with specified model"
    )
    llm_group.add_argument(
        "--llm-gguf",
        default="",
        metavar="PATH",
        help="Use local GGUF model file"
    )
    llm_group.add_argument(
        "--llm-model",
        default="",
        metavar="REPO",
        help="Use HuggingFace model for prompt generation"
    )
