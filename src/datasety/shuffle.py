"""Generate random captions by shuffling text groups."""

import sys
from pathlib import Path

from datasety.common import get_image_files


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
            req = urllib.request.Request(
                value,
                headers={"User-Agent": "datasety"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
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
    image_files = get_image_files(input_dir, formats, recursive=args.recursive)

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


def register_parser(subparsers):
    """Register the shuffle subcommand."""
    shuffle_parser = subparsers.add_parser(
        "shuffle", help="Generate random captions by shuffling text groups"
    )
    shuffle_parser.add_argument(
        "--input", "-i", required=True, help="Input directory containing images"
    )
    shuffle_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for caption .txt files"
    )
    shuffle_parser.add_argument(
        "--group",
        "-g",
        action="append",
        help="Text group with variants separated by | (e.g., 'Hello.|Hey!|Bonjour.'). "
        "Use multiple --group flags for multiple groups.",
    )
    shuffle_parser.add_argument(
        "--separator", default=" ", help="Separator between groups (default: space)"
    )
    shuffle_parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    shuffle_parser.add_argument(
        "--dry-run", action="store_true", help="Preview generated captions without writing files"
    )
    shuffle_parser.add_argument(
        "--show-distribution",
        action="store_true",
        help="Show caption distribution after generation",
    )
    shuffle_parser.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search input directory recursively for images",
    )
    shuffle_parser.set_defaults(func=cmd_shuffle)
