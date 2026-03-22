"""Resize and crop images to target resolution."""

import sys
from pathlib import Path

from PIL import Image

from datasety.common import _resolve_io_mode, get_image_files, get_save_kwargs


def _resolution_from_megapixel(megapixel, aspect_ratio):
    """Calculate width x height from megapixel count and aspect ratio string.

    Args:
        megapixel: Total megapixels (e.g., 0.5, 1.0)
        aspect_ratio: Ratio string like "16:9", "1:1", "3:2"

    Returns:
        (width, height) rounded to multiples of 8
    """
    import math

    w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
    total_pixels = megapixel * 1_000_000
    # width/height = w_ratio/h_ratio, width * height = total_pixels
    height = math.sqrt(total_pixels * h_ratio / w_ratio)
    width = height * w_ratio / h_ratio
    # Round to nearest multiple of 8
    width = round(width / 8) * 8
    height = round(height / 8) * 8
    return int(width), int(height)


def _resolution_from_megapixel_and_dims(megapixel, orig_width, orig_height):
    """Calculate target width x height from megapixel count, preserving original aspect ratio.

    Args:
        megapixel: Total megapixels (e.g., 0.5, 1.0)
        orig_width: Original image width
        orig_height: Original image height

    Returns:
        (width, height) rounded to multiples of 8
    """
    import math

    total_pixels = megapixel * 1_000_000
    aspect = orig_width / orig_height
    height = math.sqrt(total_pixels / aspect)
    width = height * aspect
    width = round(width / 8) * 8
    height = round(height / 8) * 8
    return int(width), int(height)


def calculate_resize_and_crop(
    orig_width: int, orig_height: int, target_width: int, target_height: int, crop_position: str
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

        # Parse input formats
        formats = [f.strip() for f in args.input_format.split(",")]
        image_files = get_image_files(input_dir, formats, recursive=args.recursive)

        if not image_files:
            print(f"No images found in '{input_dir}' with formats: {formats}")
            sys.exit(0)

    # Parse resolution
    has_resolution = getattr(args, "resolution", None)
    has_megapixel = getattr(args, "megapixel", None)
    has_aspect = getattr(args, "aspect_ratio", None)

    # when True, resolution is calculated per image from its native aspect ratio
    per_image_megapixel = False
    width = height = 0

    if has_resolution and has_megapixel:
        print("Error: Cannot use both --resolution and --megapixel. Choose one.")
        sys.exit(1)
    elif has_megapixel:
        if has_aspect:
            try:
                width, height = _resolution_from_megapixel(args.megapixel, args.aspect_ratio)
            except (ValueError, ZeroDivisionError):
                print(f"Error: Invalid --aspect-ratio '{args.aspect_ratio}'. Use W:H (e.g., 16:9)")
                sys.exit(1)
        else:
            per_image_megapixel = True
    elif has_resolution:
        try:
            width, height = map(int, args.resolution.lower().split("x"))
        except ValueError:
            print(
                f"Error: Invalid resolution '{args.resolution}'. Use WIDTHxHEIGHT (e.g., 768x1024)"
            )
            sys.exit(1)
    else:
        print("Error: Either --resolution or --megapixel is required.")
        sys.exit(1)

    print(f"Found {len(image_files)} images")
    if per_image_megapixel:
        print(f"Target: {args.megapixel} megapixel (per-image aspect ratio)")
    else:
        print(f"Target resolution: {width}x{height}")
    print(f"Crop position: {args.crop_position}")
    print(f"Output format: {args.output_format}")

    dry_run = args.dry_run
    if dry_run:
        print("=== DRY RUN (no files will be written) ===")

    print("-" * 50)

    processed = 0
    skipped = 0
    total = len(image_files)

    for idx, img_path in enumerate(image_files, start=1):
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                orig_w, orig_h = img.size

                if per_image_megapixel:
                    width, height = _resolution_from_megapixel_and_dims(
                        args.megapixel, orig_w, orig_h
                    )

                # Skip only if image is too small in both dimensions (truly undersized)
                if orig_w < width and orig_h < height:
                    print(f"[SKIP] {img_path.name}: {orig_w}x{orig_h} < {width}x{height}")
                    skipped += 1
                    continue

                if per_image_megapixel:
                    # No crop needed — target matches native aspect ratio
                    img_final = img.resize((width, height), Image.LANCZOS)
                else:
                    # Calculate resize and crop
                    (new_w, new_h), crop_box = calculate_resize_and_crop(
                        orig_w, orig_h, width, height, args.crop_position
                    )
                    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
                    img_final = img_resized.crop(crop_box)

                # Determine output filename
                if is_single and output_path:
                    out_path = output_path
                elif args.output_name_numbers:
                    out_path = output_dir / f"{processed + 1}.{args.output_format}"
                else:
                    out_path = output_dir / f"{img_path.stem}.{args.output_format}"

                # Save with quality settings
                save_kw = get_save_kwargs(args.output_format)
                if not dry_run:
                    img_final.save(out_path, **save_kw)

                print(
                    f"[{idx}/{total}] [OK] {img_path.name} ({orig_w}x{orig_h}) "
                    f"-> {out_path.name} ({width}x{height})"
                )
                processed += 1

        except Exception as e:
            print(f"[{idx}/{total}] [ERROR] {img_path.name}: {e}")
            skipped += 1

    print("-" * 50)
    print(f"Done! Processed: {processed}, Skipped: {skipped}")
    if dry_run and processed > 0:
        print(f"\nRun without --dry-run to process {processed} images.")


def register_parser(subparsers):
    """Register the resize subcommand."""
    resize_parser = subparsers.add_parser(
        "resize", help="Resize and crop images to target resolution"
    )
    resize_parser.add_argument(
        "--input", "-i", default="", help="Input directory containing images"
    )
    resize_parser.add_argument(
        "--output", "-o", default="", help="Output directory for processed images"
    )
    resize_parser.add_argument(
        "--input-image", default=None, help="Single input image path (alternative to --input dir)"
    )
    resize_parser.add_argument(
        "--output-image", default=None, help="Single output image path (use with --input-image)"
    )
    resize_parser.add_argument(
        "--resolution",
        "-r",
        default=None,
        help="Target resolution as WIDTHxHEIGHT (e.g., 768x1024)",
    )
    resize_parser.add_argument(
        "--megapixel",
        type=float,
        default=None,
        help="Target megapixel count (e.g., 0.5, 1.0). "
        "Without --aspect-ratio, preserves each image's native ratio.",
    )
    resize_parser.add_argument(
        "--aspect-ratio",
        default=None,
        help="Aspect ratio as W:H (e.g., 1:1, 16:9, 3:2). Use with --megapixel.",
    )
    resize_parser.add_argument(
        "--crop-position",
        choices=["top", "center", "bottom", "left", "right"],
        default="center",
        help="Position to keep when cropping (default: center)",
    )
    resize_parser.add_argument(
        "--input-format",
        default="jpg,jpeg,png,webp",
        help="Comma-separated input formats (default: jpg,jpeg,png,webp)",
    )
    resize_parser.add_argument(
        "--output-format",
        choices=["jpg", "png", "webp"],
        default="jpg",
        help="Output image format (default: jpg)",
    )
    resize_parser.add_argument(
        "--output-name-numbers",
        action="store_true",
        help="Rename output files to sequential numbers (1.jpg, 2.jpg, ...)",
    )
    resize_parser.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search input directory recursively for images",
    )
    resize_parser.add_argument(
        "--dry-run", action="store_true", help="Preview resize operations without writing files"
    )
    resize_parser.set_defaults(func=cmd_resize)
