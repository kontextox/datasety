#!/usr/bin/env python3
"""
datasety - dataset preparation: resize, align, caption, shuffle, synthetic, mask.

Usage:
    datasety resize --input ./in --output ./out --resolution 768x1024 --crop-position top
    datasety align --target ./target --control ./control --dry-run
    datasety caption --input ./in --output ./out --trigger-word "[trigger]" --florence-2-large
    datasety shuffle --input ./in --output ./out --group "Hello.|Hey!" --group "World.|Earth!"
    datasety synthetic --input ./in --output ./out --prompt "add a winter hat"
    datasety mask --input ./in --output ./masks --keywords "face,hair" --model clipseg
"""

import argparse
import sys
from pathlib import Path

from PIL import Image


def get_image_files(input_dir: Path, formats: list[str]) -> list[Path]:
    """Find all images matching the specified formats."""
    files = []
    for fmt in formats:
        fmt = fmt.lower().strip()
        files.extend(input_dir.glob(f"*.{fmt}"))
        files.extend(input_dir.glob(f"*.{fmt.upper()}"))
    return sorted(set(files))


def calculate_resize_and_crop(
    orig_width: int, orig_height: int,
    target_width: int, target_height: int,
    crop_position: str
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """
    Calculate resize dimensions and crop box.

    Args:
        crop_position: Where to position the crop window (what to keep).
                      'top' keeps top, 'right' keeps right, etc.

    Returns:
        (new_width, new_height), (left, top, right, bottom)
    """
    target_ratio = target_width / target_height
    orig_ratio = orig_width / orig_height

    if orig_ratio > target_ratio:
        # Image is wider - resize by height, crop width
        new_height = target_height
        new_width = int(orig_width * (target_height / orig_height))
    else:
        # Image is taller - resize by width, crop height
        new_width = target_width
        new_height = int(orig_height * (target_width / orig_width))

    # Calculate crop box based on position (what to keep)
    if crop_position == "center":
        left = (new_width - target_width) // 2
        top = (new_height - target_height) // 2
    elif crop_position == "top":
        left = (new_width - target_width) // 2
        top = 0
    elif crop_position == "bottom":
        left = (new_width - target_width) // 2
        top = new_height - target_height
    elif crop_position == "left":
        left = 0
        top = (new_height - target_height) // 2
    elif crop_position == "right":
        left = new_width - target_width
        top = (new_height - target_height) // 2
    else:
        raise ValueError(f"Invalid crop position: {crop_position}")

    right = left + target_width
    bottom = top + target_height

    return (new_width, new_height), (left, top, right, bottom)


def cmd_resize(args):
    """Execute the resize command."""
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse resolution
    try:
        width, height = map(int, args.resolution.lower().split("x"))
    except ValueError:
        print(f"Error: Invalid resolution '{args.resolution}'. Use WIDTHxHEIGHT (e.g., 768x1024)")
        sys.exit(1)

    # Parse input formats
    formats = [f.strip() for f in args.input_format.split(",")]

    # Get image files
    image_files = get_image_files(input_dir, formats)

    if not image_files:
        print(f"No images found in '{input_dir}' with formats: {formats}")
        sys.exit(0)

    print(f"Found {len(image_files)} images")
    print(f"Target resolution: {width}x{height}")
    print(f"Crop position: {args.crop_position}")
    print(f"Output format: {args.output_format}")
    print("-" * 50)

    processed = 0
    skipped = 0

    for idx, img_path in enumerate(image_files, start=1):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                orig_w, orig_h = img.size

                # Skip if image is too small
                if orig_w < width or orig_h < height:
                    print(f"[SKIP] {img_path.name}: {orig_w}x{orig_h} < {width}x{height}")
                    skipped += 1
                    continue

                # Calculate resize and crop
                (new_w, new_h), crop_box = calculate_resize_and_crop(
                    orig_w, orig_h, width, height, args.crop_position
                )

                # Resize
                img_resized = img.resize((new_w, new_h), Image.LANCZOS)

                # Crop
                img_cropped = img_resized.crop(crop_box)

                # Determine output filename
                if args.output_name_numbers:
                    out_name = f"{processed + 1}.{args.output_format}"
                else:
                    out_name = f"{img_path.stem}.{args.output_format}"

                out_path = output_dir / out_name

                # Save with quality settings
                save_kwargs = {}
                if args.output_format.lower() in ("jpg", "jpeg"):
                    save_kwargs["quality"] = 95
                    save_kwargs["optimize"] = True
                elif args.output_format.lower() == "webp":
                    save_kwargs["quality"] = 95
                elif args.output_format.lower() == "png":
                    save_kwargs["optimize"] = True

                img_cropped.save(out_path, **save_kwargs)

                print(f"[OK] {img_path.name} ({orig_w}x{orig_h}) -> {out_name} ({width}x{height})")
                processed += 1

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")
            skipped += 1

    print("-" * 50)
    print(f"Done! Processed: {processed}, Skipped: {skipped}")


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


def cmd_caption(args):
    """Execute the caption command."""
    # Lazy import for faster CLI startup when not using caption
    try:
        import torch
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: pip install torch transformers")
        sys.exit(1)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine model: --model takes priority, then --florence-2-base/--florence-2-large flags
    if args.model:
        model_name = args.model
    elif args.florence_2_base:
        model_name = "microsoft/Florence-2-base"
    else:
        model_name = "microsoft/Florence-2-large"

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

    # Find images (common formats)
    formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    image_files = get_image_files(input_dir, formats)

    if not image_files:
        print(f"No images found in '{input_dir}'")
        sys.exit(0)

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
                            print("Warning: beam search failed, falling back to greedy decoding")
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
                print("Hint: if all images fail, this is likely a model/transformers issue.")
                break

    print("-" * 50)
    print(f"Done! Processed: {processed} images")


def cmd_align(args):
    """Align control/target image pairs for training compatibility."""
    target_dir = Path(args.target)
    control_dir = Path(args.control)

    if not target_dir.exists():
        print(f"Error: Target directory '{target_dir}' does not exist.")
        sys.exit(1)
    if not control_dir.exists():
        print(f"Error: Control directory '{control_dir}' does not exist.")
        sys.exit(1)

    formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    target_files = get_image_files(target_dir, formats)
    control_files = get_image_files(control_dir, formats)

    if not target_files:
        print(f"No images found in '{target_dir}'")
        sys.exit(0)

    # Build lookup by stem for control images
    control_by_stem: dict[str, Path] = {}
    for cf in control_files:
        control_by_stem[cf.stem] = cf

    multiple_32 = args.multiple_of
    dry_run = args.dry_run
    out_format = args.output_format

    if dry_run:
        print("=== DRY RUN (no changes will be made) ===")

    print(f"Target: {target_dir} ({len(target_files)} images)")
    print(f"Control: {control_dir} ({len(control_files)} images)")
    print(f"Align dimensions to multiple of: {multiple_32}")
    print("-" * 50)

    fixed = 0
    skipped = 0
    missing = 0
    already_ok = 0
    errors = 0
    issues: list[str] = []

    for tf in target_files:
        cf = control_by_stem.get(tf.stem)
        if cf is None:
            issues.append(f"[MISSING] {tf.name}: no matching control image")
            missing += 1
            continue

        try:
            with Image.open(tf) as t_img, Image.open(cf) as c_img:
                t_w, t_h = t_img.size
                c_w, c_h = c_img.size

                # Round target dimensions down to multiple
                aligned_w = (t_w // multiple_32) * multiple_32
                aligned_h = (t_h // multiple_32) * multiple_32

                if aligned_w == 0 or aligned_h == 0:
                    issues.append(
                        f"[SKIP] {tf.name}: {t_w}x{t_h} too small for multiple of {multiple_32}"
                    )
                    skipped += 1
                    continue

                needs_target_fix = (t_w != aligned_w or t_h != aligned_h)
                needs_control_fix = (c_w != aligned_w or c_h != aligned_h)
                needs_format_fix_t = (out_format and tf.suffix.lstrip(".").lower() != out_format)
                needs_format_fix_c = (out_format and cf.suffix.lstrip(".").lower() != out_format)

                no_fix_needed = (
                    not needs_target_fix and not needs_control_fix
                    and not needs_format_fix_t and not needs_format_fix_c
                )
                if no_fix_needed:
                    already_ok += 1
                    continue

                detail_parts = []
                if needs_target_fix:
                    detail_parts.append(f"target {t_w}x{t_h} -> {aligned_w}x{aligned_h}")
                if needs_control_fix:
                    detail_parts.append(f"control {c_w}x{c_h} -> {aligned_w}x{aligned_h}")
                if needs_format_fix_t:
                    detail_parts.append(f"target format -> .{out_format}")
                if needs_format_fix_c:
                    detail_parts.append(f"control format -> .{out_format}")

                detail = ", ".join(detail_parts)
                print(f"[FIX] {tf.stem}: {detail}")

                if not dry_run:
                    save_kwargs = {}
                    if out_format in ("jpg", "jpeg"):
                        save_kwargs["quality"] = 95
                        save_kwargs["optimize"] = True
                    elif out_format == "webp":
                        save_kwargs["quality"] = 95
                    elif out_format == "png":
                        save_kwargs["optimize"] = True

                    if needs_target_fix or needs_format_fix_t:
                        t_rgb = t_img.convert("RGB")
                        if needs_target_fix:
                            # Center crop to aligned dimensions
                            left = (t_w - aligned_w) // 2
                            top = (t_h - aligned_h) // 2
                            t_rgb = t_rgb.crop((left, top, left + aligned_w, top + aligned_h))
                        new_t_path = tf.parent / f"{tf.stem}.{out_format}" if out_format else tf
                        t_rgb.save(new_t_path, **save_kwargs)
                        # Remove old file if format changed
                        if out_format and new_t_path != tf:
                            tf.unlink()

                    if needs_control_fix or needs_format_fix_c:
                        c_rgb = c_img.convert("RGB")
                        # Resize control to match aligned target dimensions
                        if needs_control_fix:
                            c_rgb = c_rgb.resize((aligned_w, aligned_h), Image.LANCZOS)
                        new_c_path = cf.parent / f"{cf.stem}.{out_format}" if out_format else cf
                        c_rgb.save(new_c_path, **save_kwargs)
                        if out_format and new_c_path != cf:
                            cf.unlink()

                fixed += 1

        except Exception as e:
            issues.append(f"[ERROR] {tf.name}: {e}")
            errors += 1

    # Check for orphan control images (no matching target)
    target_stems = {tf.stem for tf in target_files}
    for cf in control_files:
        if cf.stem not in target_stems:
            issues.append(f"[ORPHAN] {cf.name}: no matching target image")

    print("-" * 50)
    if issues:
        print("Issues:")
        for issue in issues:
            print(f"  {issue}")
        print("-" * 50)

    print(f"Already OK: {already_ok}")
    print(f"Fixed: {fixed}")
    print(f"Missing control: {missing}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")

    if dry_run and fixed > 0:
        print(f"\nRun without --dry-run to apply {fixed} fixes.")


def _parse_group(value: str) -> list[str]:
    """Parse a group value into a list of variants.

    Supports:
      - URL (https://... or http://...) -> fetch lines
      - File path (existing .txt file) -> read lines
      - Inline "A|B|C" -> split by pipe
    """
    if value.startswith(("https://", "http://")):
        import urllib.request
        try:
            with urllib.request.urlopen(value, timeout=15) as resp:
                text = resp.read().decode("utf-8")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                print(f"  Loaded {len(lines)} variants from URL: {value}")
                return lines
            print(f"Warning: URL returned no lines: {value}")
            return []
        except Exception as e:
            print(f"Error fetching URL '{value}': {e}")
            sys.exit(1)

    path = Path(value)
    if path.is_file():
        text = path.read_text(encoding="utf-8")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            print(f"  Loaded {len(lines)} variants from file: {value}")
            return lines
        print(f"Warning: File has no lines: {value}")
        return []

    # Inline: split by pipe
    return [v.strip() for v in value.split("|") if v.strip()]


def cmd_shuffle(args):
    """Generate random captions by picking one variant from each group."""
    import random

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.group:
        print("Error: At least one --group is required.")
        sys.exit(1)

    # Parse groups: file, URL, or inline "variant1|variant2|variant3"
    groups: list[list[str]] = []
    for g in args.group:
        variants = _parse_group(g)
        if not variants:
            print(f"Error: Empty group: '{g}'")
            sys.exit(1)
        groups.append(variants)

    if args.seed is not None:
        random.seed(args.seed)

    separator = args.separator

    # Find images
    formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    image_files = get_image_files(input_dir, formats)

    if not image_files:
        print(f"No images found in '{input_dir}'")
        sys.exit(0)

    print(f"Found {len(image_files)} images")
    print(f"Groups: {len(groups)}")
    for i, g in enumerate(groups, 1):
        print(f"  Group {i}: {len(g)} variants — {g}")
    total_combinations = 1
    for g in groups:
        total_combinations *= len(g)
    print(f"Total possible combinations: {total_combinations}")
    print("-" * 50)

    dry_run = args.dry_run
    if dry_run:
        print("=== DRY RUN (no files will be written) ===")

    processed = 0
    captions_used: dict[str, int] = {}

    for img_path in image_files:
        # Pick one random variant from each group
        parts = [random.choice(g) for g in groups]
        caption = separator.join(parts)

        captions_used[caption] = captions_used.get(caption, 0) + 1

        if dry_run:
            print(f"  {img_path.name} -> {caption}")
        else:
            caption_path = output_dir / f"{img_path.stem}.txt"
            caption_path.write_text(caption)
            print(f"[OK] {img_path.name}")

        processed += 1

    print("-" * 50)
    print(f"Done! {processed} captions generated")
    print(f"Unique captions: {len(captions_used)}")

    if args.show_distribution:
        print("\nCaption distribution:")
        for caption, count in sorted(captions_used.items(), key=lambda x: -x[1]):
            print(f"  {count}x: {caption}")


_MODEL_VRAM_GB = {
    "qwen": 32,
    "flux-kontext": 33,
    "flux2-klein": 8,
    "sdxl": 7,
    "hunyuan": 48,
}


def _detect_model_family(model_name: str) -> str:
    """Detect model family from model name/path."""
    name_lower = model_name.lower()
    if "kontext" in name_lower:
        return "flux-kontext"
    if "flux.2" in name_lower or "flux2" in name_lower or "klein" in name_lower:
        return "flux2-klein"
    if "stable-diffusion-xl" in name_lower or "sdxl" in name_lower:
        return "sdxl"
    if "hunyuan" in name_lower:
        return "hunyuan"
    return "qwen"


def _load_synthetic_pipeline(model_name, family, device, torch_dtype, gguf_path, cpu_offload):
    """Load the appropriate diffusion pipeline for the model family."""
    if family == "qwen":
        from diffusers import QwenImageEditPlusPipeline
        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import GGUFQuantizationConfig, QwenVLTransformer2DModel
            transformer = QwenVLTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            )
            kwargs["transformer"] = transformer
        pipeline = QwenImageEditPlusPipeline.from_pretrained(model_name, **kwargs)

    elif family == "flux-kontext":
        from diffusers import FluxKontextPipeline
        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            )
            kwargs["transformer"] = transformer
        pipeline = FluxKontextPipeline.from_pretrained(model_name, **kwargs)

    elif family == "flux2-klein":
        kwargs = {"torch_dtype": torch_dtype}
        try:
            from diffusers import Flux2KleinPipeline
            if gguf_path:
                from diffusers import Flux2Transformer2DModel, GGUFQuantizationConfig
                transformer = Flux2Transformer2DModel.from_single_file(
                    gguf_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                )
                kwargs["transformer"] = transformer
            pipeline = Flux2KleinPipeline.from_pretrained(model_name, **kwargs)
        except ImportError:
            from diffusers import FluxImg2ImgPipeline
            if gguf_path:
                from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
                transformer = FluxTransformer2DModel.from_single_file(
                    gguf_path,
                    quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
                )
                kwargs["transformer"] = transformer
            # FLUX.2-klein only has 5 components; provide None for missing ones
            kwargs.setdefault("text_encoder_2", None)
            kwargs.setdefault("tokenizer_2", None)
            kwargs.setdefault("image_encoder", None)
            kwargs.setdefault("feature_extractor", None)
            pipeline = FluxImg2ImgPipeline.from_pretrained(model_name, **kwargs)

    elif family == "sdxl":
        from diffusers import StableDiffusionXLImg2ImgPipeline
        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_name, torch_dtype=torch_dtype
        )

    elif family == "hunyuan":
        from diffusers import HunyuanImagePipeline
        kwargs = {"torch_dtype": torch_dtype}
        if gguf_path:
            from diffusers import GGUFQuantizationConfig, HunyuanVideo3DTransformerModel
            transformer = HunyuanVideo3DTransformerModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch_dtype),
            )
            kwargs["transformer"] = transformer
        pipeline = HunyuanImagePipeline.from_pretrained(model_name, **kwargs)

    else:
        raise ValueError(f"Unknown model family: {family}")

    if cpu_offload:
        pipeline.enable_model_cpu_offload()
        print("Model CPU offload enabled")
    else:
        pipeline.to(device)
    pipeline.set_progress_bar_config(disable=False)
    return pipeline


def _run_synthetic_pipeline(pipeline, family, image, args, device, cpu_offload):
    """Run the pipeline with family-specific parameter mapping."""
    import torch

    gen_device = "cpu" if cpu_offload else device

    gen_kwargs = {
        "prompt": args.prompt,
        "num_inference_steps": args.steps,
        "num_images_per_prompt": args.num_images,
    }

    if args.seed is not None:
        gen_kwargs["generator"] = torch.Generator(
            device=gen_device
        ).manual_seed(args.seed)

    if family == "qwen":
        gen_kwargs["image"] = [image]
        gen_kwargs["negative_prompt"] = args.negative_prompt
        gen_kwargs["guidance_scale"] = args.cfg_scale
        gen_kwargs["true_cfg_scale"] = args.true_cfg_scale

    elif family == "flux-kontext":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale

    elif family == "flux2-klein":
        gen_kwargs["guidance_scale"] = args.cfg_scale
        if "Img2Img" in type(pipeline).__name__:
            gen_kwargs["image"] = image
            gen_kwargs["strength"] = args.strength
        else:
            gen_kwargs["image"] = [image]

    elif family == "sdxl":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale
        gen_kwargs["strength"] = args.strength
        if args.negative_prompt and args.negative_prompt.strip():
            gen_kwargs["negative_prompt"] = args.negative_prompt

    elif family == "hunyuan":
        gen_kwargs["image"] = image
        gen_kwargs["guidance_scale"] = args.cfg_scale

    with torch.inference_mode():
        output = pipeline(**gen_kwargs)

    return output


def cmd_synthetic(args):
    """Execute the synthetic image generation command."""
    # Lazy import for faster CLI startup
    try:
        import torch
    except ImportError:
        print("Error: PyTorch not installed.")
        print("Run: pip install 'datasety[synthetic]'")
        sys.exit(1)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    # Detect model family
    family = _detect_model_family(args.model)

    print(f"Loading model: {args.model} (family: {family})")
    print(f"Device: {device}")

    # Auto-detect cpu_offload if not explicitly set
    if device == "cuda" and not args.cpu_offload:
        free_vram_gb = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
        needed_gb = _MODEL_VRAM_GB.get(family, 16)
        if free_vram_gb < needed_gb + 2:
            cpu_offload = True
            print(f"Auto-enabling CPU offload: {free_vram_gb:.1f} GB free, "
                  f"model needs ~{needed_gb} GB")
        else:
            cpu_offload = False
    else:
        cpu_offload = args.cpu_offload

    try:
        pipeline = _load_synthetic_pipeline(
            args.model, family, device, torch_dtype,
            getattr(args, "gguf", None), cpu_offload,
        )
    except ImportError as e:
        print(f"Error: Missing dependency for {family} pipeline: {e}")
        print("Make sure you have the latest diffusers: pip install -U diffusers")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Inject fine-tuned weights if specified (Qwen-specific)
    if args.weights:
        if family != "qwen":
            print("Warning: --weights is only supported for Qwen models, ignoring.")
        else:
            import gc

            try:
                from huggingface_hub import hf_hub_download
                from safetensors.torch import load_file
            except ImportError:
                print("Error: huggingface_hub and safetensors are required for --weights.")
                print("Run: pip install huggingface_hub safetensors")
                sys.exit(1)

            # Parse repo_id:filename
            if ":" not in args.weights:
                print("Error: --weights must be in 'repo_id:filename' format")
                print("Example: Phr00t/Qwen-Image-Edit-Rapid-AIO:v23/model.safetensors")
                sys.exit(1)

            repo_id, filename = args.weights.split(":", 1)
            print(f"Downloading weights: {repo_id} / {filename}")
            weight_path = hf_hub_download(repo_id, filename)

            print("Loading weight file...")
            state_dict = load_file(weight_path)

            # Sort weights by key prefix into component dicts
            transformer_weights = {}
            vae_weights = {}
            text_encoder_weights = {}

            for key, value in state_dict.items():
                if key.startswith(("model.diffusion_model.", "transformer.")):
                    for prefix in ("model.diffusion_model.", "transformer."):
                        if key.startswith(prefix):
                            transformer_weights[key[len(prefix):]] = value
                            break
                elif key.startswith(("first_stage_model.", "vae.")):
                    for prefix in ("first_stage_model.", "vae."):
                        if key.startswith(prefix):
                            vae_weights[key[len(prefix):]] = value
                            break
                elif "text_encoder" in key or "conditioner" in key:
                    text_encoder_weights[key] = value

            if transformer_weights:
                print(f"Injecting {len(transformer_weights)} transformer weights")
                pipeline.transformer.load_state_dict(transformer_weights, strict=False)

            if vae_weights:
                print(f"Injecting {len(vae_weights)} VAE weights")
                pipeline.vae.load_state_dict(vae_weights, strict=False)

            if text_encoder_weights:
                print(f"Injecting {len(text_encoder_weights)} text encoder weights")
                pipeline.text_encoder.load_state_dict(
                    text_encoder_weights, strict=False
                )

            del state_dict, transformer_weights, vae_weights, text_encoder_weights
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            print("Weights injected successfully")

    # Find images
    formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    image_files = get_image_files(input_dir, formats)

    if not image_files:
        print(f"No images found in '{input_dir}'")
        sys.exit(0)

    print(f"Found {len(image_files)} images")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, CFG: {args.cfg_scale}")
    if family == "qwen":
        print(f"True CFG: {args.true_cfg_scale}")
    print("-" * 50)

    processed = 0

    out_ext = args.output_format.lower()

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB").copy()

            output = _run_synthetic_pipeline(pipeline, family, image, args, device, cpu_offload)

            # Save output image(s)
            for idx, out_img in enumerate(output.images):
                if args.num_images > 1:
                    out_name = f"{img_path.stem}_{idx + 1}.{out_ext}"
                else:
                    out_name = f"{img_path.stem}.{out_ext}"

                out_path = output_dir / out_name
                out_img.save(out_path)

            print(f"[OK] {img_path.name} -> {len(output.images)} image(s)")
            processed += 1

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")

    if processed == 0 and image_files:
        print("Error: All images failed to process.")
        sys.exit(1)


def _load_mask_model_sam3(device, torch_dtype):
    """Load SAM 3 for text-prompted segmentation."""
    from transformers import Sam3Model, Sam3Processor

    primary = "facebook/sam3"
    fallback = "jetjodh/sam3"
    try:
        processor = Sam3Processor.from_pretrained(primary)
        model = Sam3Model.from_pretrained(
            primary, torch_dtype=torch_dtype,
        ).to(device).eval()
        return model, processor
    except (OSError, Exception) as e:
        print(f"Could not load {primary} ({e}), falling back to {fallback}")
    processor = Sam3Processor.from_pretrained(fallback)
    model = Sam3Model.from_pretrained(
        fallback, torch_dtype=torch_dtype,
    ).to(device).eval()
    return model, processor


def _load_mask_model_grounded_sam2(device, torch_dtype):
    """Load Grounding DINO + SAM 2 for grounded segmentation."""
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    # Load in float32 to avoid dtype mismatches with processor outputs
    dino_id = "IDEA-Research/grounding-dino-base"
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
        dino_id,
    ).to(device).eval()

    sam2_id = "facebook/sam2-hiera-large"
    sam2_processor = Sam2Processor.from_pretrained(sam2_id)
    sam2_model = Sam2Model.from_pretrained(
        sam2_id,
    ).to(device).eval()
    return (dino_model, dino_processor, sam2_model, sam2_processor)


def _load_mask_model_clipseg(device, torch_dtype):
    """Load CLIPSeg for text-based segmentation."""
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor
    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained(
        "CIDAS/clipseg-rd64-refined",
    ).to(device).eval()
    return model, processor


def _segment_sam3(model, processor, image, keywords, threshold, device):
    """Run SAM 3 segmentation for each keyword and return combined mask."""
    import numpy as np
    import torch

    w, h = image.size
    combined = np.zeros((h, w), dtype=np.uint8)

    for keyword in keywords:
        inputs = processor(images=image, text=keyword, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs, threshold=threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        for mask in results["masks"]:
            m = mask.cpu().numpy()
            combined = np.maximum(combined, m.astype(np.uint8) * 255)

    return combined


def _segment_grounded_sam2(models, image, keywords, threshold, device):
    """Run Grounding DINO + SAM 2 segmentation."""
    import numpy as np
    import torch

    dino_model, dino_processor, sam2_model, sam2_processor = models
    w, h = image.size
    combined = np.zeros((h, w), dtype=np.uint8)

    # Grounding DINO: detect boxes for all keywords at once
    text = ". ".join(keywords) + "."
    inputs = dino_processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu()
    if len(boxes) == 0:
        return combined

    # SAM 2: segment within each detected box
    sam2_inputs = sam2_processor(
        images=image, input_boxes=[boxes.tolist()], return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        sam2_outputs = sam2_model(**sam2_inputs)
    masks = sam2_processor.post_process_masks(
        sam2_outputs.pred_masks,
        sam2_inputs["original_sizes"],
        sam2_inputs["reshaped_input_sizes"],
    )
    for mask_set in masks:
        for mask in mask_set:
            m = mask.cpu().numpy()
            if m.ndim == 3:
                m = m[0]
            combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)

    return combined


def _segment_clipseg(model, processor, image, keywords, threshold, device):
    """Run CLIPSeg segmentation for each keyword."""
    import numpy as np
    import torch

    w, h = image.size
    combined = np.zeros((h, w), dtype=np.float32)

    for keyword in keywords:
        inputs = processor(
            text=[keyword], images=[image], return_tensors="pt", padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0]  # (H_clip, W_clip)
        probs = torch.sigmoid(logits).cpu().numpy()
        # Resize to full resolution
        prob_img = Image.fromarray(probs).resize((w, h), Image.BILINEAR)
        combined = np.maximum(combined, np.array(prob_img))

    return ((combined >= threshold) * 255).astype(np.uint8)


def cmd_mask(args):
    """Generate binary masks from images using text keywords."""
    try:
        import numpy as np
        import torch
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: pip install 'datasety[mask]'")
        sys.exit(1)

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.naming == "folder" else input_dir

    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if args.naming == "folder":
        output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    # Parse keywords
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if not keywords:
        print("Error: No valid keywords provided.")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Found {len(keywords)} keywords: {keywords}")
    print(f"Threshold: {args.threshold}")
    if args.dry_run:
        print("=== DRY RUN (no files will be saved) ===")
    print("-" * 50)

    # Load model
    print("Loading segmentation model...")
    try:
        if args.model == "sam3":
            models = _load_mask_model_sam3(device, torch_dtype)
        elif args.model == "grounded-sam2":
            models = _load_mask_model_grounded_sam2(device, torch_dtype)
        elif args.model == "clipseg":
            models = _load_mask_model_clipseg(device, torch_dtype)
        else:
            print(f"Error: Unknown model '{args.model}'. Use: sam3, grounded-sam2, clipseg")
            sys.exit(1)
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Run: pip install 'datasety[mask]'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Find images
    formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
    image_files = get_image_files(input_dir, formats)

    if not image_files:
        print(f"No images found in '{input_dir}'")
        sys.exit(0)

    print(f"Found {len(image_files)} images")
    print("-" * 50)

    processed = 0
    out_fmt = args.output_format

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                w, h = image.size

                # Run segmentation
                if args.model == "sam3":
                    mask_array = _segment_sam3(
                        models[0], models[1], image, keywords, args.threshold, device,
                    )
                elif args.model == "grounded-sam2":
                    mask_array = _segment_grounded_sam2(
                        models, image, keywords, args.threshold, device,
                    )
                elif args.model == "clipseg":
                    mask_array = _segment_clipseg(
                        models[0], models[1], image, keywords, args.threshold, device,
                    )

            # Apply padding (dilation)
            if args.padding > 0:
                from PIL import ImageFilter
                mask_img = Image.fromarray(mask_array, mode="L")
                mask_img = mask_img.filter(
                    ImageFilter.MaxFilter(size=args.padding * 2 + 1)
                )
                mask_array = np.array(mask_img)

            # Apply blur
            if args.blur > 0:
                from PIL import ImageFilter
                mask_img = Image.fromarray(mask_array, mode="L")
                mask_img = mask_img.filter(
                    ImageFilter.GaussianBlur(radius=args.blur)
                )
                mask_array = np.array(mask_img)

            # Invert if requested
            if args.invert:
                mask_array = 255 - mask_array

            # Determine output path
            if args.naming == "folder":
                out_path = output_dir / f"{img_path.stem}.{out_fmt}"
            else:
                out_path = input_dir / f"{img_path.stem}_mask.{out_fmt}"

            pixel_count = int(np.sum(mask_array > 127))
            coverage = pixel_count / (w * h) * 100

            if args.dry_run:
                print(f"  {img_path.name}: {len(keywords)} keywords, "
                      f"{coverage:.1f}% coverage -> {out_path.name}")
            else:
                mask_img = Image.fromarray(mask_array, mode="L")
                mask_img.save(out_path)
                print(f"[OK] {img_path.name} -> {out_path.name} ({coverage:.1f}% coverage)")

            processed += 1

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")


def main():
    from datasety import __version__

    parser = argparse.ArgumentParser(
        prog="datasety",
        description="CLI tool for dataset preparation: image resizing and captioning."
    )
    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # === RESIZE command ===
    resize_parser = subparsers.add_parser(
        "resize",
        help="Resize and crop images to target resolution"
    )
    resize_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    resize_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for processed images"
    )
    resize_parser.add_argument(
        "--resolution", "-r",
        required=True,
        help="Target resolution as WIDTHxHEIGHT (e.g., 768x1024)"
    )
    resize_parser.add_argument(
        "--crop-position",
        choices=["top", "center", "bottom", "left", "right"],
        default="center",
        help="Position to keep when cropping (default: center)"
    )
    resize_parser.add_argument(
        "--input-format",
        default="jpg,jpeg,png,webp",
        help="Comma-separated input formats (default: jpg,jpeg,png,webp)"
    )
    resize_parser.add_argument(
        "--output-format",
        choices=["jpg", "png", "webp"],
        default="jpg",
        help="Output image format (default: jpg)"
    )
    resize_parser.add_argument(
        "--output-name-numbers",
        action="store_true",
        help="Rename output files to sequential numbers (1.jpg, 2.jpg, ...)"
    )
    resize_parser.set_defaults(func=cmd_resize)

    # === CAPTION command ===
    caption_parser = subparsers.add_parser(
        "caption",
        help="Generate captions for images using Florence-2"
    )
    caption_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    caption_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for caption text files"
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
        help="Florence-2 prompt (default: <MORE_DETAILED_CAPTION>)"
    )

    caption_parser.add_argument(
        "--model",
        default="",
        help="HuggingFace model name (overrides --florence-2-base/--florence-2-large)"
    )
    caption_parser.add_argument(
        "--num-beams",
        type=int,
        default=3,
        help="Beam search width (default: 3, use 1 for greedy decoding)"
    )
    model_group = caption_parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--florence-2-base",
        action="store_true",
        help="Use Florence-2-base model (0.23B params, faster)"
    )
    model_group.add_argument(
        "--florence-2-large",
        action="store_true",
        help="Use Florence-2-large model (0.77B params, more accurate) [default]"
    )
    caption_parser.set_defaults(func=cmd_caption)

    # === ALIGN command ===
    align_parser = subparsers.add_parser(
        "align",
        help="Align control/target image pairs (match dimensions, format, multiple of 32)"
    )
    align_parser.add_argument(
        "--target", "-t",
        required=True,
        help="Target images directory"
    )
    align_parser.add_argument(
        "--control", "-c",
        required=True,
        help="Control images directory"
    )
    align_parser.add_argument(
        "--multiple-of",
        type=int,
        default=32,
        help="Align dimensions to this multiple (default: 32)"
    )
    align_parser.add_argument(
        "--output-format",
        choices=["jpg", "png", "webp", ""],
        default="",
        help="Convert all images to this format (default: keep original)"
    )
    align_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    align_parser.set_defaults(func=cmd_align)

    # === SHUFFLE command ===
    shuffle_parser = subparsers.add_parser(
        "shuffle",
        help="Generate random captions by shuffling text groups"
    )
    shuffle_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    shuffle_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for caption .txt files"
    )
    shuffle_parser.add_argument(
        "--group", "-g",
        action="append",
        help="Text group with variants separated by | (e.g., 'Hello.|Hey!|Bonjour.'). "
        "Use multiple --group flags for multiple groups."
    )
    shuffle_parser.add_argument(
        "--separator",
        default=" ",
        help="Separator between groups (default: space)"
    )
    shuffle_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    shuffle_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview generated captions without writing files"
    )
    shuffle_parser.add_argument(
        "--show-distribution",
        action="store_true",
        help="Show caption distribution after generation"
    )
    shuffle_parser.set_defaults(func=cmd_shuffle)

    # === SYNTHETIC command ===
    synthetic_parser = subparsers.add_parser(
        "synthetic",
        help="Generate synthetic images using image editing models"
    )
    synthetic_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    synthetic_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for generated images"
    )
    synthetic_parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Edit prompt (e.g., 'add a winter hat to the person')"
    )
    synthetic_parser.add_argument(
        "--model",
        default="Qwen/Qwen-Image-Edit-2511",
        help="Model to use (default: Qwen/Qwen-Image-Edit-2511)"
    )
    synthetic_parser.add_argument(
        "--weights",
        default=None,
        help="Fine-tuned weights as 'repo_id:filename' "
        "(e.g., 'Phr00t/Qwen-Image-Edit-Rapid-AIO:v23/model.safetensors')"
    )
    synthetic_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU)"
    )
    synthetic_parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Force CPU offload (auto-detected by default based on available VRAM)"
    )
    synthetic_parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of inference steps (default: 40)"
    )
    synthetic_parser.add_argument(
        "--cfg-scale",
        type=float,
        default=1.0,
        help="Guidance scale (default: 1.0)"
    )
    synthetic_parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="True CFG scale (default: 4.0)"
    )
    synthetic_parser.add_argument(
        "--negative-prompt",
        default=" ",
        help="Negative prompt (default: ' ')"
    )
    synthetic_parser.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images to generate per input (default: 1)"
    )
    synthetic_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    synthetic_parser.add_argument(
        "--gguf",
        default=None,
        help="Path or URL to GGUF file for quantized transformer loading"
    )
    synthetic_parser.add_argument(
        "--strength",
        type=float,
        default=0.7,
        help="Img2img strength for SDXL/FLUX.2 (0.0-1.0, default: 0.7)"
    )
    synthetic_parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)"
    )
    synthetic_parser.set_defaults(func=cmd_synthetic)

    # === MASK command ===
    mask_parser = subparsers.add_parser(
        "mask",
        help="Generate binary masks from images using text keywords"
    )
    mask_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input directory containing images"
    )
    mask_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output directory for mask images"
    )
    mask_parser.add_argument(
        "--keywords", "-k",
        required=True,
        help="Comma-separated keywords to segment (e.g., 'face,hair,hat')"
    )
    mask_parser.add_argument(
        "--model",
        choices=["sam3", "grounded-sam2", "clipseg"],
        default="sam3",
        help="Segmentation model (default: sam3)"
    )
    mask_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU)"
    )
    mask_parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detection (0.0-1.0, default: 0.3)"
    )
    mask_parser.add_argument(
        "--padding",
        type=int,
        default=0,
        help="Pixels to expand mask by (dilation, default: 0)"
    )
    mask_parser.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Gaussian blur radius for mask edges (0=sharp, default: 0)"
    )
    mask_parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert mask (black=ROI, white=ignore)"
    )
    mask_parser.add_argument(
        "--naming",
        choices=["folder", "suffix"],
        default="folder",
        help="Output naming: 'folder' (same name in output dir) or "
        "'suffix' (_mask suffix in input dir) (default: folder)"
    )
    mask_parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)"
    )
    mask_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview detections without saving masks"
    )
    mask_parser.set_defaults(func=cmd_mask)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
