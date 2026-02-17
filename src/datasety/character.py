"""Generate character datasets using LLM prompts + FLUX.2-klein text-to-image."""

import sys
from pathlib import Path

from datasety.common import _resolve_gguf_path, resolve_device
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
                    line = line[idx + len(sep) :]
                    break
        line = line.strip().strip('"').strip("'")
        if line:
            lines.append(line)
    return lines


def cmd_character(args):
    """Execute the character dataset generation command."""
    # Validate reference images (optional)
    ref_paths = [Path(r) for r in args.reference] if args.reference else []
    for rp in ref_paths:
        if not rp.exists():
            print(f"Error: Reference image not found: {rp}")
            sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dry_run = args.dry_run
    if dry_run:
        print("=== DRY RUN (no files will be written) ===")

    # ── Step 1: Generate or load prompts ──
    if args.prompts_file:
        prompts_path = Path(args.prompts_file)
        if not prompts_path.exists():
            print(f"Error: Prompts file not found: {prompts_path}")
            sys.exit(1)
        prompts = [line.strip() for line in prompts_path.read_text().splitlines() if line.strip()]
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

        user_prompt = _build_user_prompt(args.character_description, args.style, args.num_images)
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
    if not dry_run:
        prompts_out.write_text("\n".join(prompts))

    if args.prompts_only:
        if not dry_run:
            print(f"Prompts saved to {prompts_out}")
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p}")
        return

    if dry_run:
        for i, p in enumerate(prompts, 1):
            print(f"  {i}. {p}")
        out_ext = args.output_format
        print(f"\nPlanned output ({len(prompts)} images):")
        for i in range(1, len(prompts) + 1):
            print(f"  {output_dir / f'{i:04d}.{out_ext}'}")
        print(f"\nRun without --dry-run to generate {len(prompts)} images.")
        return

    # ── Step 2: Generate images ──

    if args.image_api:
        # ── Cloud API path ──
        from datasety.llm import _generate_image_via_api, resolve_llm_api_config

        api_key, base_url, model = resolve_llm_api_config(args.model or None)
        if not api_key:
            print("Error: OPENAI_API_KEY environment variable is required for --image-api")
            sys.exit(1)

        print(f"Using image API: {base_url}")
        print(f"Model: {model}")
        print(f"Generating {len(prompts)} images...")
        print("-" * 50)

        processed = 0
        out_ext = args.output_format

        for i, prompt in enumerate(prompts, 1):
            try:
                result = _generate_image_via_api(
                    prompt,
                    api_key,
                    base_url,
                    model,
                    seed=args.seed,
                )

                out_path = output_dir / f"{i:04d}.{out_ext}"
                result.save(out_path)

                # Save prompt alongside image
                (output_dir / f"{i:04d}.txt").write_text(prompt)

                print(f"[OK] {i}/{len(prompts)}: {prompt[:80]}...")
                processed += 1

            except Exception as e:
                print(f"[ERROR] {i}/{len(prompts)}: {e}")

    else:
        # ── Local pipeline path ──
        try:
            import torch
        except ImportError:
            print("Error: PyTorch not installed.")
            print("Run: pip install 'datasety[character]'")
            sys.exit(1)

        device = resolve_device(args.device)
        torch_dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32

        print(f"Loading model: {args.model}")
        print(f"Device: {device}")

        gguf_path = _resolve_gguf_path(getattr(args, "gguf", None))

        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig

            transformer = Flux2Transformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                torch_dtype=torch_dtype,
                config=args.model,
                subfolder="transformer",
            )
            kwargs["transformer"] = transformer

        try:
            from diffusers import Flux2KleinPipeline
        except ImportError:
            import subprocess

            print(
                "Flux2KleinPipeline not found, upgrading diffusers from "
                "official HuggingFace repo..."
            )
            print("Installing: git+https://github.com/huggingface/diffusers.git")
            subprocess.check_call(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-q",
                    "git+https://github.com/huggingface/diffusers.git",
                ]
            )
            import importlib

            for _key in [k for k in sys.modules if k.startswith("diffusers")]:
                del sys.modules[_key]
            importlib.invalidate_caches()
            from diffusers import Flux2KleinPipeline

        pipeline = Flux2KleinPipeline.from_pretrained(args.model, **kwargs)
        pipeline.enable_model_cpu_offload()
        pipeline.set_progress_bar_config(disable=False)

        print(f"Generating {len(prompts)} images...")
        if ref_paths:
            print(f"Reference images: {len(ref_paths)} (for prompt context only)")
        print("-" * 50)

        processed = 0
        out_ext = args.output_format

        gen_kwargs = {
            "height": args.height,
            "width": args.width,
            "num_inference_steps": args.steps,
            "guidance_scale": args.cfg_scale,
        }

        if args.seed is not None:
            gen_kwargs["generator"] = torch.Generator(device="cpu").manual_seed(args.seed)

        for i, prompt in enumerate(prompts, 1):
            try:
                with torch.inference_mode():
                    output = pipeline(prompt=prompt, **gen_kwargs)

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
        "character", help="Generate character datasets using text-to-image (FLUX.2-klein)"
    )

    # Reference images (optional, for prompt context)
    char_parser.add_argument(
        "--reference",
        "-r",
        nargs="+",
        default=[],
        help="Reference face image(s) for prompt context (optional)",
    )
    char_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for generated dataset"
    )
    char_parser.add_argument(
        "--num-images",
        "-n",
        type=int,
        default=10,
        help="Number of images to generate (default: 10)",
    )

    # Model selection
    char_parser.add_argument(
        "--model",
        default="black-forest-labs/FLUX.2-klein-4B",
        help="Base model for image generation (default: black-forest-labs/FLUX.2-klein-4B)",
    )
    char_parser.add_argument(
        "--gguf", default=None, help="Path or URL to GGUF file for quantized transformer loading"
    )

    # LLM backend selection
    add_llm_arguments(char_parser)

    # Prompt options
    char_parser.add_argument(
        "--character-description",
        default="",
        help="Text description of the character (important for identity consistency)",
    )
    char_parser.add_argument(
        "--style", default="", help="Style guidance (e.g., 'photorealistic', 'anime')"
    )
    char_parser.add_argument(
        "--prompts-only", action="store_true", help="Only generate prompts, skip image generation"
    )
    char_parser.add_argument(
        "--prompts-file", default="", help="Load prompts from file instead of generating with LLM"
    )

    # Generation settings
    char_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU/MPS)",
    )
    char_parser.add_argument(
        "--steps", type=int, default=4, help="Number of inference steps (default: 4)"
    )
    char_parser.add_argument(
        "--cfg-scale",
        type=float,
        default=4.0,
        help="Guidance scale (default: 4.0, recommended for FLUX.2-klein)",
    )
    char_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    char_parser.add_argument(
        "--height", type=int, default=1024, help="Output image height (default: 1024)"
    )
    char_parser.add_argument(
        "--width", type=int, default=1024, help="Output image width (default: 1024)"
    )
    char_parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)",
    )
    char_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview prompts and output paths without generating images",
    )
    char_parser.add_argument(
        "--image-api",
        action="store_true",
        help="Use OpenAI-compatible API for image generation (needs OPENAI_API_KEY)",
    )

    char_parser.set_defaults(func=cmd_character)
