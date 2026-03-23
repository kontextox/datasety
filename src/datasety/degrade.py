"""Create degraded versions of images for upscale/enhance training."""

import random
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

from datasety.common import _resolve_io_mode, get_image_files

# ── Degradation registry ──

DEGRADATION_TYPES = [
    "lowres",
    "oversharpen",
    "noise",
    "blur",
    "jpeg",
    "motion-blur",
    "pixelate",
    "color-bands",
    "upscale-sim",
]


def _degrade_lowres(image, intensity):
    """Bilinear downscale + nearest upscale. intensity=1.0 → 16x downscale."""
    w, h = image.size
    factor = max(2, int(2 + intensity * 14))  # 2x .. 16x
    small = image.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return small.resize((w, h), Image.NEAREST)


def _degrade_oversharpen(image, intensity):
    """Unsharp mask with aggressive settings. intensity=1.0 → percent=1000."""
    percent = int(100 + intensity * 900)
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=percent, threshold=0))


def _degrade_noise(image, intensity):
    """Add Gaussian noise. intensity=1.0 → 50% noise blend."""
    w, h = image.size
    noise_bytes = random.randbytes(w * h * 3)
    noise_img = Image.frombytes("RGB", (w, h), noise_bytes)
    alpha = min(0.5, intensity * 0.5)
    return Image.blend(image, noise_img, alpha)


def _degrade_blur(image, intensity):
    """Gaussian blur. intensity=1.0 → radius=3."""
    radius = max(0.5, intensity * 3.0)
    return image.filter(ImageFilter.GaussianBlur(radius=radius))


def _degrade_jpeg(image, intensity):
    """JPEG compression artifacts. intensity=1.0 → quality=5."""
    quality = max(1, int(95 - intensity * 90))  # 95 → 5
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


def _degrade_motion_blur(image, intensity):
    """Horizontal motion blur using sequential box blurs for large radii."""
    # Pillow Kernel is limited to small sizes; use repeated horizontal box blur
    radius = max(1, int(1 + intensity * 15))  # 1..16
    # Apply horizontal blur by resizing: squash horizontally then stretch back
    w, h = image.size
    squash_w = max(1, w // (radius + 1))
    blurred = image.resize((squash_w, h), Image.BILINEAR)
    return blurred.resize((w, h), Image.BILINEAR)


def _degrade_pixelate(image, intensity):
    """Nearest-neighbor pixelation. intensity=1.0 → 16x down+up."""
    w, h = image.size
    factor = max(2, int(2 + intensity * 14))
    small = image.resize((max(1, w // factor), max(1, h // factor)), Image.NEAREST)
    return small.resize((w, h), Image.NEAREST)


def _degrade_color_bands(image, intensity):
    """Posterization (reduce color depth). intensity=1.0 → 3-bit."""
    bits = max(1, int(8 - intensity * 5))  # 8 → 3
    return ImageOps.posterize(image, bits)


def _degrade_upscale_sim(image, intensity):
    """Simulate smooth AI-upscaler look: bilinear down + Lanczos up."""
    w, h = image.size
    factor = max(2, int(2 + intensity * 6))  # 2x .. 8x
    small = image.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
    return small.resize((w, h), Image.LANCZOS)


_DEGRADATION_FUNCS = {
    "lowres": _degrade_lowres,
    "oversharpen": _degrade_oversharpen,
    "noise": _degrade_noise,
    "blur": _degrade_blur,
    "jpeg": _degrade_jpeg,
    "motion-blur": _degrade_motion_blur,
    "pixelate": _degrade_pixelate,
    "color-bands": _degrade_color_bands,
    "upscale-sim": _degrade_upscale_sim,
}


def apply_degradations(image, types, intensity=0.5, chain=False, intensity_range=None):
    """Apply degradation(s) to an image.

    Args:
        image: PIL Image (RGB).
        types: List of degradation type names. "random" picks a random type.
        intensity: Float 0.0-1.0 (used when intensity_range is None).
        chain: If True, apply all types sequentially. If False, pick one.
        intensity_range: Optional (min, max) tuple. When set, each step in
            the chain gets an independently randomized intensity.

    Returns:
        (degraded_image, steps) where steps is a list of
        (type_name, intensity_used) tuples describing what was applied.
    """
    # Resolve "random"
    resolved = []
    for t in types:
        if t == "random":
            resolved.append(random.choice(DEGRADATION_TYPES))
        else:
            resolved.append(t)

    if not chain:
        resolved = [random.choice(resolved)]

    result = image.copy()
    steps = []
    for t in resolved:
        func = _DEGRADATION_FUNCS.get(t)
        if func is None:
            raise ValueError(f"Unknown degradation type: {t}")
        if intensity_range is not None:
            step_intensity = random.uniform(*intensity_range)
        else:
            step_intensity = intensity
        result = func(result, step_intensity)
        steps.append((t, step_intensity))

    return result, steps


def cmd_degrade(args):
    """Execute the degrade command."""
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

    # Validate types
    types = args.type or ["random"]
    for t in types:
        if t != "random" and t not in _DEGRADATION_FUNCS:
            print(
                f"Error: Unknown degradation type '{t}'. "
                f"Available: {', '.join(DEGRADATION_TYPES)}, random"
            )
            sys.exit(1)

    # Handle intensity / intensity-range
    intensity_value = max(0.0, min(1.0, args.intensity))
    if args.intensity_range:
        try:
            parts = args.intensity_range.split("-")
            intensity_min = float(parts[0])
            intensity_max = float(parts[1])
            intensity_min = max(0.0, min(1.0, intensity_min))
            intensity_max = max(0.0, min(1.0, intensity_max))
        except (ValueError, IndexError):
            print(
                f"Error: Invalid intensity range '{args.intensity_range}'. "
                "Use MIN-MAX (e.g., 0.3-0.8)"
            )
            sys.exit(1)
        intensity_range = (intensity_min, intensity_max)
    else:
        intensity_range = None

    if args.seed is not None:
        random.seed(args.seed)

    paired = args.paired

    # Find images
    if not is_single:
        formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
        image_files = get_image_files(input_dir, formats, recursive=args.recursive)

        if not image_files:
            print(f"No images found in '{input_dir}'")
            sys.exit(0)

    # Set up paired directories
    if paired and not is_single:
        control_dir = output_dir / "control"
        target_dir = output_dir / "target"
        control_dir.mkdir(parents=True, exist_ok=True)
        target_dir.mkdir(parents=True, exist_ok=True)

    num_variants = args.num_variants

    dry_run = args.dry_run
    if dry_run:
        print("=== DRY RUN (no files will be written) ===")

    print(f"Found {len(image_files)} images")
    print(f"Degradation types: {types}")
    print(f"Chain: {args.chain}")
    if intensity_range:
        print(f"Intensity range: {intensity_range[0]}-{intensity_range[1]}")
    else:
        print(f"Intensity: {intensity_value}")
    if num_variants > 1:
        print(f"Variants per image: {num_variants}")
    if paired:
        print("Paired mode: control/ (degraded) + target/ (original)")
    print("-" * 50)

    processed = 0
    total = len(image_files)
    out_ext = args.output_format

    def _process_one_degrade(idx, img_path):
        """Process a single image through degradation. Returns (idx, logs) or raises."""
        if getattr(args, "skip_existing", False) and not is_single:
            out_check = output_dir / f"{img_path.stem}.{out_ext}"
            if out_check.exists():
                return idx, "skip", []

        with Image.open(img_path) as img:
            image = img.convert("RGB")

            variant_logs = []
            count = 0

            for variant_idx in range(num_variants):
                degraded, steps = apply_degradations(
                    image,
                    types,
                    intensity=intensity_value,
                    chain=args.chain,
                    intensity_range=intensity_range,
                )

                if num_variants > 1:
                    stem = f"{img_path.stem}_{variant_idx + 1}"
                else:
                    stem = img_path.stem

                if not dry_run:
                    if is_single and output_path:
                        degraded.save(output_path)
                    elif paired:
                        ctrl_path = control_dir / f"{stem}.{out_ext}"
                        tgt_path = target_dir / f"{stem}.{out_ext}"
                        degraded.save(ctrl_path)
                        image.save(tgt_path)
                    else:
                        o_path = output_dir / f"{stem}.{out_ext}"
                        degraded.save(o_path)

                count += 1
                steps_str = " > ".join(f"{name}:{intens:.2f}" for name, intens in steps)
                variant_logs.append((f"{stem}.{out_ext}", steps_str))

        return idx, "ok", variant_logs

    if args.workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_process_one_degrade, idx, img_path): (idx, img_path)
                for idx, img_path in enumerate(image_files, 1)
            }
            for future in as_completed(futures):
                idx, img_path = futures[future]
                try:
                    _, status, variant_logs = future.result()
                    if status == "skip":
                        print(f"[{idx}/{total}] [SKIP] {img_path.name} (output exists)")
                        continue
                    processed += len(variant_logs)
                    if num_variants > 1:
                        print(f"[{idx}/{total}] [OK] {img_path.name} -> {num_variants} variants")
                        for fname, steps_str in variant_logs:
                            print(f"     {fname} ({steps_str})")
                    else:
                        fname, steps_str = variant_logs[0]
                        print(f"[{idx}/{total}] [OK] {img_path.name} -> {fname} ({steps_str})")
                except Exception as e:
                    print(f"[{idx}/{total}] [ERROR] {img_path.name}: {e}")
    else:
        for idx, img_path in enumerate(image_files, 1):
            try:
                _, status, variant_logs = _process_one_degrade(idx, img_path)
                if status == "skip":
                    print(f"[{idx}/{total}] [SKIP] {img_path.name} (output exists)")
                    continue
                processed += len(variant_logs)
                if num_variants > 1:
                    print(f"[{idx}/{total}] [OK] {img_path.name} -> {num_variants} variants")
                    for fname, steps_str in variant_logs:
                        print(f"     {fname} ({steps_str})")
                else:
                    fname, steps_str = variant_logs[0]
                    print(f"[{idx}/{total}] [OK] {img_path.name} -> {fname} ({steps_str})")
            except Exception as e:
                print(f"[{idx}/{total}] [ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")
    if dry_run and processed > 0:
        print(f"\nRun without --dry-run to process {processed} images.")


def register_parser(subparsers):
    """Register the degrade subcommand."""
    degrade_parser = subparsers.add_parser(
        "degrade", help="Create degraded image versions for upscale/enhance training"
    )
    degrade_parser.add_argument(
        "--input", "-i", default="", help="Input directory containing images"
    )
    degrade_parser.add_argument(
        "--output", "-o", default="", help="Output directory for degraded images"
    )
    degrade_parser.add_argument(
        "--input-image", default=None, help="Single input image path (alternative to --input dir)"
    )
    degrade_parser.add_argument(
        "--output-image", default=None, help="Single output image path (use with --input-image)"
    )
    degrade_parser.add_argument(
        "--type",
        "-t",
        action="append",
        default=None,
        help=f"Degradation type(s): {', '.join(DEGRADATION_TYPES)}, random. "
        "Can be specified multiple times. (default: random)",
    )
    degrade_parser.add_argument(
        "--intensity", type=float, default=0.5, help="Global intensity 0.0-1.0 (default: 0.5)"
    )
    degrade_parser.add_argument(
        "--intensity-range",
        default=None,
        help="Random intensity range MIN-MAX (e.g., 0.3-0.8). Overrides --intensity.",
    )
    degrade_parser.add_argument(
        "--chain",
        action="store_true",
        help="Apply multiple --type degradations sequentially (default: pick one)",
    )
    degrade_parser.add_argument(
        "--num-variants",
        type=int,
        default=1,
        help="Number of degraded variants per input image (default: 1). "
        "Each variant gets fresh random degradations and intensity.",
    )
    degrade_parser.add_argument(
        "--paired",
        action="store_true",
        help="Create control/ (degraded) + target/ (original) subdirs",
    )
    degrade_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    degrade_parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)",
    )
    degrade_parser.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search input directory recursively for images",
    )
    degrade_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing (default: 1)",
    )
    degrade_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar instead of per-file output",
    )
    degrade_parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose output file already exists",
    )
    degrade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview degradation operations without writing files",
    )
    degrade_parser.set_defaults(func=cmd_degrade)
