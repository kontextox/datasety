#!/usr/bin/env python3
"""
datasety - CLI tool for dataset preparation: resize, caption, and synthetic generation.

Usage:
    datasety resize --input ./in --output ./out --resolution 768x1024 --crop-position top
    datasety caption --input ./in --output ./out --trigger-word "[trigger]" --florence-2-large
    datasety synthetic --input ./in --output ./out --prompt "add a winter hat"
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


def cmd_caption(args):
    """Execute the caption command."""
    # Lazy import for faster CLI startup when not using caption
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
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

    # Determine model (base flag takes priority since large is default)
    if args.florence_2_base:
        model_name = "microsoft/Florence-2-base"
    else:
        model_name = "microsoft/Florence-2-large"

    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading model: {model_name}")
    print(f"Device: {device}")

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
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

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")

                inputs = processor(
                    text=args.prompt,
                    images=img,
                    return_tensors="pt"
                ).to(device, torch_dtype)

                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=inputs["input_ids"],
                        pixel_values=inputs["pixel_values"],
                        max_new_tokens=1024,
                        num_beams=3,
                        do_sample=False
                    )

                generated_text = processor.batch_decode(
                    generated_ids, skip_special_tokens=False
                )[0]

                parsed = processor.post_process_generation(
                    generated_text,
                    task=args.prompt,
                    image_size=(img.width, img.height)
                )

                caption = parsed.get(args.prompt, "")

                # Prepend trigger word if specified
                if args.trigger_word:
                    caption = f"{args.trigger_word} {caption}"

                # Save caption
                caption_path = output_dir / f"{img_path.stem}.txt"
                caption_path.write_text(caption.strip())

                print(f"[OK] {img_path.name}")
                print(f"     {caption[:100]}{'...' if len(caption) > 100 else ''}")
                processed += 1

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")


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
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    # Import the correct pipeline based on model
    try:
        from diffusers import QwenImageEditPlusPipeline
        pipeline_class = QwenImageEditPlusPipeline
    except ImportError:
        print("Error: QwenImageEditPlusPipeline not found.")
        print("Make sure you have the latest diffusers: pip install -U diffusers")
        sys.exit(1)

    print(f"Loading model: {args.model}")
    print(f"Device: {device}")

    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    try:
        pipeline = pipeline_class.from_pretrained(
            args.model,
            torch_dtype=torch_dtype
        )
        pipeline.to(device)
        pipeline.set_progress_bar_config(disable=False)
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
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, CFG: {args.cfg_scale}, True CFG: {args.true_cfg_scale}")
    print("-" * 50)

    processed = 0

    for img_path in image_files:
        try:
            image = Image.open(img_path).convert("RGB")

            # Set up generation parameters
            gen_kwargs = {
                "image": [image],
                "prompt": args.prompt,
                "negative_prompt": args.negative_prompt,
                "num_inference_steps": args.steps,
                "guidance_scale": args.cfg_scale,
                "true_cfg_scale": args.true_cfg_scale,
                "num_images_per_prompt": args.num_images,
            }

            # Add seed if specified
            if args.seed is not None:
                gen_kwargs["generator"] = torch.manual_seed(args.seed)

            with torch.inference_mode():
                output = pipeline(**gen_kwargs)

            # Save output image(s)
            for idx, out_img in enumerate(output.images):
                if args.num_images > 1:
                    out_name = f"{img_path.stem}_{idx + 1}.png"
                else:
                    out_name = f"{img_path.stem}.png"

                out_path = output_dir / out_name
                out_img.save(out_path)

            print(f"[OK] {img_path.name} -> {len(output.images)} image(s)")
            processed += 1

        except Exception as e:
            print(f"[ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")


def main():
    parser = argparse.ArgumentParser(
        prog="datasety",
        description="CLI tool for dataset preparation: image resizing and captioning."
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
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run model on (default: cpu)"
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
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help="Device to run model on (default: cuda)"
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
    synthetic_parser.set_defaults(func=cmd_synthetic)

    # Parse and execute
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
