"""Inspect dataset: statistics, caption coverage, duplicate detection."""

import csv
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

from datasety.common import get_image_files, hamming_distance, image_phash


def cmd_inspect(args):
    """Execute the inspect command."""
    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    formats = [f.strip() for f in "jpg,jpeg,png,webp,bmp,tiff".split(",")]
    image_files = get_image_files(input_dir, formats, recursive=args.recursive)

    if not image_files:
        print(f"No images found in '{input_dir}'")
        sys.exit(0)

    print(f"Scanning {len(image_files)} images...")
    print("-" * 50)

    # Collect stats
    widths = []
    heights = []
    format_counts = Counter()
    aspect_ratios = Counter()
    sizes_bytes = []
    captions_found = 0
    captions_missing = 0
    caption_lengths = []
    hashes = {}  # hash -> [paths]
    errors = 0

    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                fmt = img_path.suffix.lower().lstrip(".")
                format_counts[fmt] += 1

                # Bucket aspect ratio
                ratio = w / h
                if ratio > 1.4:
                    aspect_ratios["landscape"] += 1
                elif ratio < 0.7:
                    aspect_ratios["portrait"] += 1
                else:
                    aspect_ratios["square"] += 1

                sizes_bytes.append(img_path.stat().st_size)

                # Caption check
                caption_path = img_path.with_suffix(".txt")
                if caption_path.exists():
                    captions_found += 1
                    text = caption_path.read_text(encoding="utf-8").strip()
                    caption_lengths.append(len(text))
                else:
                    captions_missing += 1

                # Duplicate detection
                if args.duplicates:
                    phash = image_phash(img)
                    hashes.setdefault(phash, []).append(str(img_path))

        except Exception as e:
            print(f"  [ERROR] {img_path.name}: {e}")
            errors += 1

    total = len(image_files) - errors

    # ── Print report ──
    print(f"\n{'=' * 50}")
    print(f"  DATASET SUMMARY: {input_dir}")
    print(f"{'=' * 50}")
    print(f"  Images:          {total}")
    if errors:
        print(f"  Errors:          {errors}")
    print()

    # Resolution stats
    if widths:
        print("  Resolution:")
        print(f"    Min:           {min(widths)}x{min(heights)}")
        print(f"    Max:           {max(widths)}x{max(heights)}")
        avg_w = sum(widths) // len(widths)
        avg_h = sum(heights) // len(heights)
        print(f"    Average:       {avg_w}x{avg_h}")

        # Resolution distribution
        res_counter = Counter(f"{w}x{h}" for w, h in zip(widths, heights))
        unique_res = len(res_counter)
        print(f"    Unique sizes:  {unique_res}")
        if unique_res <= 10:
            for res, count in res_counter.most_common():
                print(f"      {res}: {count}")
        else:
            for res, count in res_counter.most_common(5):
                print(f"      {res}: {count}")
            print(f"      ... and {unique_res - 5} more")
        print()

    # Aspect ratio
    if aspect_ratios:
        print("  Orientation:")
        for orient in ["landscape", "square", "portrait"]:
            count = aspect_ratios.get(orient, 0)
            if count:
                pct = count / total * 100
                print(f"    {orient:12s}   {count:4d}  ({pct:.1f}%)")
        print()

    # Format breakdown
    if format_counts:
        print("  Formats:")
        for fmt, count in format_counts.most_common():
            pct = count / total * 100
            print(f"    .{fmt:5s}         {count:4d}  ({pct:.1f}%)")
        print()

    # File sizes
    if sizes_bytes:
        total_mb = sum(sizes_bytes) / (1024 * 1024)
        avg_kb = (sum(sizes_bytes) / len(sizes_bytes)) / 1024
        print(f"  Total size:      {total_mb:.1f} MB")
        print(f"  Avg file size:   {avg_kb:.1f} KB")
        print()

    # Caption coverage
    print("  Captions:")
    print(f"    With .txt:     {captions_found}/{total}")
    if captions_missing:
        print(f"    Missing:       {captions_missing}")
    if caption_lengths:
        avg_len = sum(caption_lengths) // len(caption_lengths)
        print(f"    Avg length:    {avg_len} chars")
        empty = sum(1 for cl in caption_lengths if cl == 0)
        if empty:
            print(f"    Empty:         {empty}")
    print()

    # Duplicates
    if args.duplicates:
        # Find near-duplicates (hamming distance <= threshold)
        dup_groups = []
        hash_list = list(hashes.items())
        seen = set()
        for i, (h1, paths1) in enumerate(hash_list):
            if h1 in seen:
                continue
            # Exact hash duplicates
            if len(paths1) > 1:
                dup_groups.append(paths1)
                seen.add(h1)
                continue
            # Near duplicates (hamming distance <= 4)
            group = list(paths1)
            for j in range(i + 1, len(hash_list)):
                h2, paths2 = hash_list[j]
                if h2 in seen:
                    continue
                if hamming_distance(h1, h2) <= 4:
                    group.extend(paths2)
                    seen.add(h2)
            if len(group) > 1:
                dup_groups.append(group)
                seen.add(h1)

        if dup_groups:
            total_dups = sum(len(g) for g in dup_groups)
            print(f"  Duplicates:      {len(dup_groups)} groups ({total_dups} images)")
            for i, group in enumerate(dup_groups[:10], 1):
                print(f"    Group {i}:")
                for p in group[:5]:
                    print(f"      {p}")
                if len(group) > 5:
                    print(f"      ... and {len(group) - 5} more")
            if len(dup_groups) > 10:
                print(f"    ... and {len(dup_groups) - 10} more groups")
        else:
            print("  Duplicates:      none found")
        print()

    print(f"{'=' * 50}")

    # Export
    if args.json:
        report = {
            "path": str(input_dir),
            "total_images": total,
            "errors": errors,
            "resolution": {
                "min": f"{min(widths)}x{min(heights)}" if widths else None,
                "max": f"{max(widths)}x{max(heights)}" if widths else None,
                "average": (
                    f"{sum(widths) // len(widths)}x{sum(heights) // len(heights)}"
                    if widths
                    else None
                ),
            },
            "formats": dict(format_counts),
            "orientation": dict(aspect_ratios),
            "total_size_mb": round(sum(sizes_bytes) / (1024 * 1024), 1) if sizes_bytes else 0,
            "captions_found": captions_found,
            "captions_missing": captions_missing,
            "avg_caption_length": (
                sum(caption_lengths) // len(caption_lengths) if caption_lengths else 0
            ),
        }
        json_path = Path(args.json)
        json_path.write_text(json.dumps(report, indent=2))
        print(f"Report saved to {json_path}")

    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "width", "height", "format", "size_kb", "has_caption"])
            for img_path in image_files:
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                    fmt = img_path.suffix.lower().lstrip(".")
                    size_kb = round(img_path.stat().st_size / 1024, 1)
                    has_cap = img_path.with_suffix(".txt").exists()
                    writer.writerow([str(img_path), w, h, fmt, size_kb, has_cap])
                except Exception:
                    pass
        print(f"CSV saved to {csv_path}")


def register_parser(subparsers):
    """Register the inspect subcommand."""
    p = subparsers.add_parser(
        "inspect", help="Show dataset statistics, caption coverage, and detect duplicates"
    )
    p.add_argument("--input", "-i", required=True, help="Input directory containing images")
    p.add_argument(
        "--duplicates",
        action="store_true",
        help="Detect duplicate/near-duplicate images using perceptual hashing",
    )
    p.add_argument(
        "--json",
        default="",
        help="Export report as JSON to this path",
    )
    p.add_argument(
        "--csv",
        default="",
        help="Export per-image data as CSV to this path",
    )
    p.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search input directory recursively for images",
    )
    p.set_defaults(func=cmd_inspect)
