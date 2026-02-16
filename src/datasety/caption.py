"""Generate captions for images using Florence-2 or OpenAI-compatible vision APIs."""

import base64
import json
import sys
from pathlib import Path

from PIL import Image

from datasety.common import _resolve_io_mode, get_image_files


def _load_caption_model_native(model_name, torch_dtype, device):
    """Load Florence-2 using native transformers support (>= 4.50)."""
    from transformers import AutoProcessor, Florence2ForConditionalGeneration

    # Map microsoft/ model names to florence-community/ for native support
    native_map = {
        "microsoft/Florence-2-base": "florence-community/Florence-2-base",
        "microsoft/Florence-2-large": "florence-community/Florence-2-large",
        "microsoft/Florence-2-base-ft": "florence-community/Florence-2-base-ft",
        "microsoft/Florence-2-large-ft": "florence-community/Florence-2-large-ft",
    }
    native_name = native_map.get(model_name, model_name)

    model = Florence2ForConditionalGeneration.from_pretrained(
        native_name, dtype=torch_dtype
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(native_name, use_fast=True)
    return model, processor


def _load_caption_model_legacy(model_name, torch_dtype, device):
    """Load Florence-2 using trust_remote_code (older transformers)."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    # Patch forced_bos_token_id on config classes so re-instantiated objects get it
    for cfg in [config] + (
        [config.text_config] if hasattr(config, "text_config") else []
    ):
        if not hasattr(cfg, "forced_bos_token_id"):
            cfg.forced_bos_token_id = 1
        cfg_cls = type(cfg)
        if not hasattr(cfg_cls, "_datasety_patched"):
            original_init = cfg_cls.__init__

            def make_patched(orig):
                def patched_init(self, *args, **kwargs):
                    orig(self, *args, **kwargs)
                    if not hasattr(self, "forced_bos_token_id"):
                        self.forced_bos_token_id = 1
                return patched_init

            cfg_cls.__init__ = make_patched(original_init)
            cfg_cls._datasety_patched = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name, config=config, torch_dtype=torch_dtype, trust_remote_code=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    return model, processor


# ── LLM API captioning ──


def _image_to_data_url(img_path):
    """Convert an image file to a base64 data URL for vision APIs."""
    suffix = img_path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp",
        ".gif": "image/gif", ".bmp": "image/bmp",
        ".tiff": "image/tiff", ".tif": "image/tiff",
    }
    mime = mime_map.get(suffix, "image/jpeg")
    data = img_path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _caption_via_api(img_path, prompt, api_key, base_url, model,
                     max_tokens, temperature):
    """Send an image to an OpenAI-compatible vision API and return the caption."""
    import urllib.request

    data_url = _image_to_data_url(img_path)

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            },
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())

    return data["choices"][0]["message"]["content"]


def _cmd_caption_llm_api(args, image_files, output_path, output_dir, is_single):
    """Caption images using an OpenAI-compatible vision API."""
    from datasety.llm import resolve_llm_api_config

    api_key, base_url, model = resolve_llm_api_config(args.model or None)
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required for --llm-api")
        sys.exit(1)
    prompt = args.prompt
    max_tokens = args.max_tokens
    temperature = args.temperature

    print(f"API: {base_url}")
    print(f"Model: {model}")
    print(f"Found {len(image_files)} images")
    print(f"Prompt: {prompt}")
    if args.trigger_word:
        print(f"Trigger word: {args.trigger_word}")
    print("-" * 50)

    processed = 0

    for img_path in image_files:
        try:
            caption = _caption_via_api(
                img_path, prompt, api_key, base_url, model,
                max_tokens, temperature,
            )
            caption = caption.strip()

            if args.trigger_word:
                caption = f"{args.trigger_word} {caption}"

            if is_single and output_path:
                caption_path = output_path
            else:
                caption_path = output_dir / f"{img_path.stem}.txt"
            caption_path.write_text(caption)

            print(f"[OK] {img_path.name}")
            print(f"     {caption[:100]}{'...' if len(caption) > 100 else ''}")
            processed += 1

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")


def cmd_caption(args):
    """Execute the caption command."""
    single_files, output_path, is_single = _resolve_io_mode(args)

    if is_single:
        image_files = single_files
        output_dir = output_path.parent if output_path else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        input_dir = Path(args.input)
        output_dir = output_path

        if not input_dir.exists():
            print(f"Error: Input directory '{input_dir}' does not exist.")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

    # Find images early (both paths need them)
    if not is_single:
        formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
        image_files = get_image_files(Path(args.input), formats)

        if not image_files:
            print(f"No images found in '{args.input}'")
            sys.exit(0)
    else:
        image_files = single_files

    # ── LLM API path ──
    if args.llm_api:
        _cmd_caption_llm_api(args, image_files, output_path, output_dir, is_single)
        return

    # ── Florence-2 path ──
    # Lazy import for faster CLI startup when not using caption
    try:
        import torch
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: pip install torch transformers")
        sys.exit(1)

    # Determine model: --model takes priority, then --florence-2-large/--florence-2-base flags
    if args.model:
        model_name = args.model
    elif args.florence_2_large:
        model_name = "microsoft/Florence-2-large"
    else:
        model_name = "microsoft/Florence-2-base"

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    try:
        # Try native transformers Florence-2 support (>= 4.50, no trust_remote_code)
        model, processor = _load_caption_model_native(model_name, torch_dtype, device)
        print("Using native Florence-2 support")
    except (ImportError, OSError, ValueError):
        # Fall back to legacy trust_remote_code approach for older transformers
        # or non-standard model repos
        try:
            model, processor = _load_caption_model_legacy(model_name, torch_dtype, device)
            print("Using legacy Florence-2 support (trust_remote_code)")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)

    print(f"Found {len(image_files)} images")
    print(f"Prompt: {args.prompt}")
    if args.trigger_word:
        print(f"Trigger word: {args.trigger_word}")
    print("-" * 50)

    processed = 0
    num_beams = args.num_beams

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")

                inputs = processor(
                    text=args.prompt,
                    images=img,
                    return_tensors="pt"
                ).to(device, torch_dtype)

                generate_kwargs = {
                    "input_ids": inputs["input_ids"],
                    "pixel_values": inputs["pixel_values"],
                    "max_new_tokens": 1024,
                    "num_beams": num_beams,
                    "do_sample": False,
                }

                with torch.no_grad():
                    try:
                        generated_ids = model.generate(**generate_kwargs)
                    except AttributeError:
                        # Beam search fails on some transformers versions due to
                        # past_key_values format changes. Fall back to greedy.
                        if num_beams > 1:
                            print("Warning: beam search failed, "
                                  "falling back to greedy decoding")
                            num_beams = 1
                            generate_kwargs["num_beams"] = 1
                            generated_ids = model.generate(**generate_kwargs)
                        else:
                            raise

                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]

                parsed = processor.post_process_generation(
                    generated_text,
                    task=args.prompt,
                    image_size=(img.width, img.height)
                )

                caption = parsed.get(args.prompt, "")

                if args.trigger_word:
                    caption = f"{args.trigger_word} {caption}"

                if is_single and output_path:
                    caption_path = output_path
                else:
                    caption_path = output_dir / f"{img_path.stem}.txt"
                caption_path.write_text(caption.strip())

                print(f"[OK] {img_path.name}")
                print(f"     {caption[:100]}{'...' if len(caption) > 100 else ''}")
                processed += 1

        except Exception as e:
            import traceback
            print(f"[ERROR] {img_path.name}: {e}")
            traceback.print_exc()
            if processed == 0:
                print("Hint: if all images fail, this is likely a "
                      "model/transformers issue.")
                break

    print("-" * 50)
    print(f"Done! Processed: {processed} images")


def register_parser(subparsers):
    """Register the caption subcommand."""
    caption_parser = subparsers.add_parser(
        "caption",
        help="Generate captions for images using Florence-2 or vision LLM APIs"
    )
    caption_parser.add_argument(
        "--input", "-i",
        default="",
        help="Input directory containing images"
    )
    caption_parser.add_argument(
        "--output", "-o",
        default="",
        help="Output directory for caption text files"
    )
    caption_parser.add_argument(
        "--input-image",
        default=None,
        help="Single input image path (alternative to --input dir)"
    )
    caption_parser.add_argument(
        "--output-caption",
        default=None,
        help="Single output .txt path (use with --input-image)"
    )
    caption_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU)"
    )
    caption_parser.add_argument(
        "--trigger-word",
        default="",
        help="Text to prepend to each caption (e.g., '[trigger]' or 'photo,')"
    )
    caption_parser.add_argument(
        "--prompt",
        default="<MORE_DETAILED_CAPTION>",
        help="Caption prompt. For Florence-2: <MORE_DETAILED_CAPTION>. "
        "For --llm-api: free-form text instruction."
    )

    caption_parser.add_argument(
        "--model",
        default="",
        help="Model name. Florence-2 HF repo, or API model ID with --llm-api "
        "(e.g., 'x-ai/grok-4.1-fast', 'gpt-4o')"
    )
    caption_parser.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Beam search width for Florence-2 (default: 3, use 1 for greedy)"
    )

    # LLM API mode
    caption_parser.add_argument(
        "--llm-api",
        action="store_true",
        help="Use OpenAI-compatible vision API (needs OPENAI_API_KEY env var). "
        "Set OPENAI_BASE_URL or OPENAI_API_BASE for non-OpenAI providers "
        "(e.g., https://openrouter.ai/api/v1)"
    )
    caption_parser.add_argument(
        "--max-tokens",
        type=int,
        default=300,
        help="Max response tokens for --llm-api (default: 300)"
    )
    caption_parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Temperature for --llm-api (default: 0.3)"
    )

    model_group = caption_parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--florence-2-base",
        action="store_true",
        help="Use Florence-2-base model (0.23B params, faster) [default]"
    )
    model_group.add_argument(
        "--florence-2-large",
        action="store_true",
        help="Use Florence-2-large model (0.77B params, more accurate)"
    )
    caption_parser.set_defaults(func=cmd_caption)
