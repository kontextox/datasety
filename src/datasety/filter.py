"""Filter dataset images by content using CLIP or NudeNet."""

import csv
import shutil
import sys
from pathlib import Path

from datasety.common import get_image_files, resolve_device

# ── CLIP backend ────────────────────────────────────────────────────────────


def _load_clip(device):
    """Load CLIP model and processor."""
    from transformers import CLIPModel, CLIPProcessor

    model_id = "openai/clip-vit-base-patch32"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device).eval()
    return model, processor


def _clip_score(image, queries, model, processor, device):
    """Return per-query cosine similarity scores for an image.

    Returns list of (query, score) tuples.
    """
    import torch

    inputs = processor(text=queries, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    # Normalised cosine similarity (0-1 range after softmax, but we use raw logits_per_image)
    logits = outputs.logits_per_image[0]  # shape: (num_queries,)
    # Convert logits to 0-1 probabilities via softmax when >1 query,
    # otherwise use sigmoid for single query
    if len(queries) == 1:
        scores = torch.sigmoid(logits / 100.0).cpu().tolist()
    else:
        # Use per-query sigmoid so each is independent
        scores = torch.sigmoid(logits / 100.0).cpu().tolist()
    return list(zip(queries, scores))


# ── NudeNet backend ─────────────────────────────────────────────────────────


def _load_nudenet():
    """Load NudeNet detector."""
    try:
        from nudenet import NudeDetector
    except ImportError:
        print("Error: nudenet is required for --model nudenet.")
        print("Run: pip install 'nudenet>=3.4.2'")
        sys.exit(1)
    return NudeDetector()


def _nudenet_score(image_path, labels, detector):
    """Return per-label max detection scores for an image.

    Returns list of (label, score) tuples.
    """
    detections = detector.detect(str(image_path))
    scores = []
    for label in labels:
        label_upper = label.upper()
        max_score = 0.0
        for det in detections:
            if det["class"] == label_upper:
                max_score = max(max_score, det["score"])
        scores.append((label, max_score))
    return scores


# ── Companion files ─────────────────────────────────────────────────────────


def _find_companions(image_path):
    """Find companion files (captions, masks) for an image."""
    stem = image_path.stem
    parent = image_path.parent
    companions = []
    for ext in (".txt", ".caption", ".json"):
        companion = parent / f"{stem}{ext}"
        if companion.exists():
            companions.append(companion)
    return companions


def _act_on_file(file_path, action, output_dir, input_root, preserve_structure):
    """Move, copy, or delete a file."""
    if action == "delete":
        file_path.unlink()
        return
    if output_dir is None:
        return
    if preserve_structure and input_root:
        rel = file_path.relative_to(input_root)
        dest = output_dir / rel
    else:
        dest = output_dir / file_path.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if action == "move":
        shutil.move(str(file_path), str(dest))
    elif action == "copy":
        shutil.copy2(str(file_path), str(dest))


# ── Main command ────────────────────────────────────────────────────────────


def cmd_filter(args):
    """Execute the filter command."""
    from PIL import Image

    # Validate args
    if not args.query and not args.labels:
        print("Error: --query or --labels is required.")
        sys.exit(1)

    if args.query and args.labels:
        print("Error: --query and --labels are mutually exclusive.")
        sys.exit(1)

    use_nudenet = args.model == "nudenet"
    use_clip = args.model == "clip"

    if args.labels and not use_nudenet:
        print("Error: --labels requires --model nudenet.")
        sys.exit(1)

    if args.query and use_nudenet:
        print("Error: --query requires --model clip. Use --labels for NudeNet.")
        sys.exit(1)

    action = args.action
    if action == "delete" and not args.confirm:
        print("Error: --confirm is required for --action delete.")
        print("This will permanently delete matching files. Add --confirm to proceed.")
        sys.exit(1)

    if action == "keep" and not args.output and not args.confirm:
        print("Error: --confirm is required for --action keep without --output.")
        print("Non-matching files will be permanently deleted. Add --confirm to proceed.")
        sys.exit(1)

    if action in ("move", "copy") and not args.output:
        print(f"Error: --output is required for --action {action}.")
        sys.exit(1)

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else None

    # Gather queries/labels
    if args.query:
        queries = [q.strip() for q in args.query.split(",") if q.strip()]
    else:
        queries = [lb.strip() for lb in args.labels.split(",") if lb.strip()]

    threshold = args.threshold
    fmt = "jpg,jpeg,png,webp,bmp,tiff"

    print(f"Model: {args.model}")
    if use_clip:
        device = resolve_device(args.device)
        print(f"Device: {device}")
    print(f"Queries: {queries}")
    print(f"Threshold: {threshold}")
    print(f"Action: {action}")
    print("-" * 50)

    # Load model
    print("Loading model...")
    if use_clip:
        model, processor = _load_clip(device)
    else:
        detector = _load_nudenet()

    # Find images
    image_files = get_image_files(input_dir, fmt.split(","), recursive=args.recursive)
    if not image_files:
        print("No images found.")
        return

    print(f"Found {len(image_files)} images")
    print("-" * 50)

    # Process
    matched = 0
    unmatched = 0
    log_rows = []

    for idx, img_path in enumerate(image_files, 1):
        try:
            if use_clip:
                image = Image.open(img_path).convert("RGB")
                scores = _clip_score(image, queries, model, processor, device)
            else:
                scores = _nudenet_score(img_path, queries, detector)

            max_query, max_score = max(scores, key=lambda x: x[1])
            is_match = max_score >= threshold
            if args.invert:
                is_match = not is_match

            status = "MATCH" if is_match else "skip"
            score_str = ", ".join(f"{q}={s:.3f}" for q, s in scores)
            print(f"[{idx}/{len(image_files)}] [{status}] {img_path.name} ({score_str})")

            if args.log:
                for q, s in scores:
                    log_rows.append(
                        {
                            "file": str(img_path),
                            "query": q,
                            "score": f"{s:.4f}",
                            "match": str(is_match),
                            "action": action if not args.dry_run else "dry-run",
                        }
                    )

            if args.dry_run:
                if is_match:
                    matched += 1
                else:
                    unmatched += 1
                continue

            companions = _find_companions(img_path)

            if action == "keep":
                if is_match:
                    matched += 1
                else:
                    unmatched += 1
                    if output_dir:
                        preserve = args.preserve_structure
                        _act_on_file(img_path, "move", output_dir, input_dir, preserve)
                        for c in companions:
                            _act_on_file(c, "move", output_dir, input_dir, preserve)
                    else:
                        img_path.unlink()
                        for c in companions:
                            c.unlink()
            else:
                if is_match:
                    matched += 1
                    _act_on_file(img_path, action, output_dir, input_dir, args.preserve_structure)
                    for c in companions:
                        _act_on_file(c, action, output_dir, input_dir, args.preserve_structure)
                else:
                    unmatched += 1

        except Exception as e:
            print(f"[{idx}/{len(image_files)}] [ERR] {img_path.name}: {e}")

    # Write log
    if args.log and log_rows:
        log_path = Path(args.log)
        with open(log_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "query", "score", "match", "action"])
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"\nLog written to {log_path}")

    print("-" * 50)
    print(f"Done! Matched: {matched}, Skipped: {unmatched}")
    if args.dry_run:
        print("(dry-run — no files were modified)")


def register_parser(subparsers):
    """Register the filter subcommand."""
    p = subparsers.add_parser(
        "filter", help="Filter dataset images by content using CLIP or NudeNet"
    )
    p.add_argument("--input", "-i", required=True, help="Input directory containing images")
    p.add_argument(
        "--output", "-o", default="", help="Output directory for matched/rejected images"
    )
    p.add_argument(
        "--query",
        "-q",
        default="",
        help="Comma-separated text queries for CLIP (e.g., 'leg,male face')",
    )
    p.add_argument(
        "--labels",
        "-l",
        default="",
        help="Comma-separated NudeNet labels (e.g., 'FEMALE_BREAST_EXPOSED,ANUS_EXPOSED')",
    )
    p.add_argument(
        "--model",
        choices=["clip", "nudenet"],
        default="clip",
        help="Detection model (default: clip)",
    )
    p.add_argument(
        "--action",
        choices=["move", "copy", "delete", "keep"],
        default="move",
        help="Action for matches: move, copy, delete, or keep (default: move)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Confidence threshold (0.0-1.0, default: 0.5)",
    )
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU/MPS)",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert match logic (act on images that do NOT match)",
    )
    p.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm destructive actions (required for --action delete)",
    )
    p.add_argument(
        "--preserve-structure",
        action="store_true",
        help="Preserve subfolder hierarchy in output (for --recursive)",
    )
    p.add_argument(
        "--log",
        default="",
        help="Write a CSV log of all decisions to this path",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Show tqdm progress bar instead of per-file output",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview detections without modifying files",
    )
    p.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search input directory recursively for images",
    )
    p.set_defaults(func=cmd_filter)
