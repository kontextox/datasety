"""Align control/target image pairs for training compatibility."""

import sys
from pathlib import Path

from PIL import Image

from datasety.common import get_image_files


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


def register_parser(subparsers):
    """Register the align subcommand."""
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
