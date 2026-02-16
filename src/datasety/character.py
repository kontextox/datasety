"""Generate character datasets from reference face images using LLM prompts + IP-Adapter."""

import sys
from pathlib import Path

from PIL import Image

from datasety.llm import (
    _create_llm_backend,
    add_llm_arguments,
)

# ── Prompt helpers ──

_SYSTEM_PROMPT = """\
You are a prompt generator for AI image generation. Generate varied, creative prompts \
that describe a specific character in different poses, settings, expressions, and outfits. \
Each prompt should be a single line describing one image. Do NOT number the prompts. \
Output one prompt per line, nothing else."""


def _build_user_prompt(description, style, num_prompts):
    """Build the user prompt for LLM prompt generation."""
    parts = [f"Generate {num_prompts} varied image prompts"]
    if description:
        parts.append(f"for this character: {description}")
    if style:
        parts.append(f"in {style} style")
    parts.append(
        "Each prompt should describe a different scene, pose, expression, "
        "lighting, or outfit. One prompt per line."
    )
    return ". ".join(parts)


def _parse_prompts(text):
    """Parse LLM output into a list of prompts."""
    lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        # Strip numbering like "1. " or "1) "
        if line and line[0].isdigit():
            for sep in [". ", ") ", ": ", "- "]:
                idx = line.find(sep)
                if idx != -1 and idx < 5:
                    line = line[idx + len(sep):]
                    break
        line = line.strip().strip('"').strip("'")
        if line:
            lines.append(line)
    return lines


def cmd_character(args):
    """Execute the character dataset generation command."""
    # Validate reference images
    if not args.reference:
        print("Error: --reference is required (one or more face images)")
        sys.exit(1)

    ref_paths = [Path(r) for r in args.reference]
    for rp in ref_paths:
        if not rp.exists():
            print(f"Error: Reference image not found: {rp}")
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Generate or load prompts ──
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"Error: Prompts file not found: {prompts_path}")
            sys.exit(1)
        prompts = [
            line.strip() for line in prompts_path.read_text().splitlines()
            if line.strip()
        ]
        print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    else:
        backend = _create_llm_backend(args)
        if backend is None:
            print("Error: An LLM backend is required. Use one of:")
            print("  --llm-api          (OpenAI-compatible API, needs OPENAI_API_KEY)")
            print("  --llm-ollama MODEL (local Ollama server)")
            print("  --llm-gguf PATH    (local GGUF model)")
            print("  --llm-model REPO   (HuggingFace model)")
            sys.exit(1)

        user_prompt = _build_user_prompt(
            args.character_description, args.style, args.num_images
        )
        print(f"Generating {args.num_images} prompts via LLM...")
        raw = backend.generate(_SYSTEM_PROMPT, user_prompt)
        prompts = _parse_prompts(raw)

        if not prompts:
            print("Error: LLM returned no valid prompts")
            print(f"Raw output:\n{raw}")
            sys.exit(1)

        print(f"Generated {len(prompts)} prompts")

    # Save prompts for reproducibility
    prompts_out = output_dir / "prompts.txt"
    prompts_out.write_text("\n".join(prompts))

    if args.prompts_only:
        print(f"Prompts saved to {prompts_out}")
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p}")
        return

    # ── Step 2: Load image generation pipeline ──
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed.")
        print("Run: pip install 'datasety[character]'")
        sys.exit(1)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")

    try:
        from diffusers import AutoPipelineForText2Image
    except ImportError:
        print("Error: diffusers is required for image generation.")
        print("Run: pip install 'datasety[character]'")
        sys.exit(1)

    pipeline = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch_dtype,
    )

    # Load IP-Adapter
    ip_adapter = args.ip_adapter
    if not ip_adapter:
        # Auto-detect based on model
        model_lower = args.model.lower()
        if "flux" in model_lower:
            ip_adapter = "XLabs-AI/flux-ip-adapter"
        else:
            ip_adapter = "h94/IP-Adapter"

    print(f"Loading IP-Adapter: {ip_adapter}")
    try:
        pipeline.load_ip_adapter(ip_adapter)
    except Exception as e:
        print(f"Error loading IP-Adapter: {e}")
        sys.exit(1)

    pipeline.set_ip_adapter_scale(args.ip_adapter_scale)

    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=False)

    # ── Step 3: Load reference face embedding ──
    ref_images = [Image.open(rp).convert("RGB") for rp in ref_paths]
    print(f"Loaded {len(ref_images)} reference image(s)")

    # ── Step 4: Generate images ──
    print(f"Generating {len(prompts)} images...")
    print("-" * 50)

    processed = 0
    out_ext = args.output_format

    gen_kwargs = {
        "num_inference_steps": args.steps,
        "guidance_scale": args.cfg_scale,
    }

    if args.seed is not None:
        gen_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)

    for i, prompt in enumerate(prompts, 1):
        try:
            output = pipeline(
                prompt=prompt,
                ip_adapter_image=ref_images,
                **gen_kwargs,
            )

            out_path = output_dir / f"{i:04d}.{out_ext}"
            output.images[0].save(out_path)

            # Save prompt alongside image
            (output_dir / f"{i:04d}.txt").write_text(prompt)

            print(f"[OK] {i}/{len(prompts)}: {prompt[:80]}...")
            processed += 1

        except Exception as e:
            print(f"[ERROR] {i}/{len(prompts)}: {e}")

    print("-" * 50)
    print(f"Done! Generated: {processed}/{len(prompts)} images")
    print(f"Output: {output_dir}")


def register_parser(subparsers):
    """Register the character subcommand."""
    char_parser = subparsers.add_parser(
        "character",
        help="Generate character datasets from reference face images"
    )

    # Reference images
    char_parser.add_argument(
        "--reference", "-r",
        nargs="+",
        required=True,
        help="Reference face image(s) for identity preservation"
    )
    char_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for generated dataset"
    )
    char_parser.add_argument(
        "--num-images", "-n",
        type=int,
        default=10,
        help="Number of images to generate (default: 10)"
    )

    # Model selection
    char_parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.1-dev",
        help="Base model for image generation (default: black-forest-labs/FLUX.1-dev)"
    )
    char_parser.add_argument(
        "--ip-adapter",
        default="",
        help="IP-Adapter model (default: auto-detect based on base model)"
    )
    char_parser.add_argument(
        "--ip-adapter-scale",
        type=float,
        default=0.6,
        help="IP-Adapter conditioning strength 0.0-1.0 (default: 0.6)"
    )

    # LLM backend selection
    add_llm_arguments(char_parser)

    # Prompt options
    char_parser.add_argument(
        "--character-description",
        default="",
        help="Text description of the character"
    )
    char_parser.add_argument(
        "--style",
        default="",
        help="Style guidance (e.g., 'photorealistic', 'anime')"
    )
    char_parser.add_argument(
        "--prompts-only",
        action="store_true",
        help="Only generate prompts, skip image generation"
    )
    char_parser.add_argument(
        "--prompts-file",
        default="",
        help="Load prompts from file instead of generating with LLM"
    )

    # Generation settings
    char_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU)"
    )
    char_parser.add_argument(
        "--steps",
        type=int,
        default=28,
        help="Number of inference steps (default: 28)"
    )
    char_parser.add_argument(
        "--cfg-scale",
        type=float,
        default=3.5,
        help="Guidance scale (default: 3.5)"
    )
    char_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    char_parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)"
    )

    char_parser.set_defaults(func=cmd_character)
