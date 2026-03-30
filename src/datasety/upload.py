"""Upload datasets and models to HuggingFace Hub.

Supports audio, image, video, document, and generic datasets, plus model uploads.
Generates HF-compliant README.md dataset cards with proper YAML frontmatter.

Usage:
    datasety upload --path ./my_dataset --repo-id user/my-dataset --type audio
    datasety upload --path ./my_lora --repo-id user/sdxl-lora --type model
    datasety upload --path ./my_dataset --type auto --dry-run

Requires: huggingface_hub>=0.20.0 (already in datasety dependencies)
"""

import csv
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Dataset type detection
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
DOCUMENT_EXTENSIONS = {".pdf"}
MODEL_EXTENSIONS = {".safetensors", ".bin", ".gguf", ".ckpt", ".pt", ".pth"}
GENERIC_EXTENSIONS = {".csv", ".json", ".jsonl", ".parquet", ".txt", ".npy"}


def detect_dataset_type(path: Path) -> str:
    """Auto-detect dataset type from directory structure.

    Returns one of: audio, image, video, document, model, generic
    """
    path = Path(path)
    if not path.exists():
        return "generic"

    all_files: list[Path] = []
    if path.is_dir():
        all_files = [p for p in path.rglob("*") if p.is_file()]
    else:
        all_files = [path]

    has_audio = any(p.suffix.lower() in AUDIO_EXTENSIONS for p in all_files)
    has_image = any(p.suffix.lower() in IMAGE_EXTENSIONS for p in all_files)
    has_video = any(p.suffix.lower() in VIDEO_EXTENSIONS for p in all_files)
    has_document = any(p.suffix.lower() in DOCUMENT_EXTENSIONS for p in all_files)
    has_model = any(p.suffix.lower() in MODEL_EXTENSIONS for p in all_files)

    # Check for HF AudioFolder structure: wavs/ + metadata.csv
    has_audio_folder = (
        path.is_dir() and (path / "wavs").exists() and (path / "metadata.csv").exists()
    )
    if has_audio or has_audio_folder:
        return "audio"

    # Check for HF ImageFolder structure: train/test/... subdirs
    if has_image:
        subdirs = {p.name for p in path.rglob("*") if p.is_dir()}
        if subdirs:
            return "image"

    if has_video:
        return "video"

    if has_document:
        return "document"

    if has_model:
        return "model"

    return "generic"


# ---------------------------------------------------------------------------
# Dataset card / README generation
# ---------------------------------------------------------------------------

_TASK_CATEGORIES = {
    "audio": ["text-to-speech"],
    "image": ["image-captioning"],
    "video": ["video-captioning"],
    "document": ["document-understanding"],
    "generic": ["other"],
    "model": [],
}

_DATASET_MODALITY = {
    "audio": ["audio"],
    "image": ["image"],
    "video": ["video"],
    "document": ["document"],
    "generic": ["text"],
    "model": [],
}


def _slugify(name: str) -> str:
    """Convert a dataset name to a URL-safe slug."""
    return name.lower().replace(" ", "-").replace("_", "-").replace(",", "")


def _detect_language_from_metadata(path: Path, dataset_type: str) -> list[str]:
    """Try to detect language from metadata files."""
    path = Path(path)

    if dataset_type == "audio":
        metadata_file = path / "metadata.csv"
        if metadata_file.exists():
            try:
                with open(metadata_file, encoding="utf-8") as f:
                    reader = csv.DictReader(f, delimiter="|")
                    for row in reader:
                        text = row.get("text", "")
                        if text:
                            return ["en"]  # default
            except Exception:
                pass

    return ["en"]  # default


def _count_examples(path: Path, dataset_type: str) -> int:
    """Count number of examples in the dataset."""
    path = Path(path)
    count = 0

    if dataset_type == "audio":
        wavs_dir = path / "wavs"
        if wavs_dir.exists():
            count = len(list(wavs_dir.glob("*.wav")) + list(wavs_dir.glob("*.mp3")))
        metadata = path / "metadata.csv"
        if metadata.exists():
            try:
                with open(metadata, encoding="utf-8") as f:
                    count = sum(1 for _ in f) - 1  # subtract header
            except Exception:
                pass

    elif dataset_type == "image":
        for ext in IMAGE_EXTENSIONS:
            count += len(list(path.rglob(f"*{ext}")))

    elif dataset_type == "video":
        for ext in VIDEO_EXTENSIONS:
            count += len(list(path.rglob(f"*{ext}")))

    elif dataset_type == "document":
        for ext in DOCUMENT_EXTENSIONS:
            count += len(list(path.rglob(f"*{ext}")))

    elif dataset_type == "generic":
        for ext in GENERIC_EXTENSIONS:
            count += len(list(path.rglob(f"*{ext}")))

    return max(count, 0)


def _size_category(n: int) -> str:
    """Return HF size category string."""
    if n < 1000:
        return f"n<{n}"
    if n < 10000:
        return f"{n // 1000}k-{n // 1000 + 1}k"
    if n < 100000:
        return f"{n // 10000}k-{n // 10000 + 1}k"
    return f"n>{n}"


def generate_dataset_card(
    path: Path,
    dataset_type: str,
    repo_name: str | None = None,
    extra_metadata: dict | None = None,
) -> str:
    """Generate a HF-compliant README.md dataset card.

    Args:
        path: Path to the dataset directory (used for auto-detection)
        dataset_type: One of audio, image, video, document, generic, model
        repo_name: Pretty name for the dataset (defaults to directory name)
        extra_metadata: Extra YAML key-value pairs to merge into frontmatter

    Returns:
        The full README.md content as a string.
    """
    path = Path(path)
    name = repo_name or path.name
    languages = _detect_language_from_metadata(path, dataset_type)
    task_cats = _TASK_CATEGORIES.get(dataset_type, ["other"])
    modalities = _DATASET_MODALITY.get(dataset_type, ["text"])
    example_count = _count_examples(path, dataset_type)
    size_cat = _size_category(example_count)

    # Build YAML frontmatter
    license_val = extra_metadata.pop("license", "mit") if extra_metadata else "mit"

    lines = ["---"]
    lines.append("annotations_creators:")
    lines.append("  - no-annotation")
    lines.append("language:")
    for lang in languages:
        lines.append(f"  - {lang}")
    lines.append("license:")
    lines.append(f"  - {license_val}")
    lines.append("multiprocess: false")
    lines.append("pretty_name: " + name)
    lines.append("size_categories:")
    lines.append(f"  - {size_cat}")
    lines.append("task_categories:")
    for tc in task_cats:
        lines.append(f"  - {tc}")
    if modalities:
        lines.append("dataset_modality:")
        for m in modalities:
            lines.append(f"  - {m}")

    # Add extra metadata
    if extra_metadata:
        for key, val in extra_metadata.items():
            if isinstance(val, list):
                lines.append(f"{key}:")
                for item in val:
                    lines.append(f"  - {item}")
            else:
                lines.append(f"{key}: {val}")

    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append("This dataset was prepared using [datasety](https://github.com/kontextox/datasety).")
    lines.append("")
    lines.append("## Dataset Summary")
    lines.append("")
    lines.append("<!-- Add a description of your dataset here. -->")
    lines.append("")
    lines.append(f"**Type:** {dataset_type}")
    if example_count > 0:
        lines.append(f"**Examples:** {example_count}")
    lines.append("")

    if dataset_type == "audio":
        lines.append("## Supported Tasks")
        lines.append("")
        lines.append("- Text-to-Speech (TTS)")
        lines.append("")
    elif dataset_type == "image":
        lines.append("## Supported Tasks")
        lines.append("")
        lines.append("- Image Captioning")
        lines.append("- Image Classification (with metadata)")
        lines.append("")
    elif dataset_type == "video":
        lines.append("## Supported Tasks")
        lines.append("")
        lines.append("- Video Captioning")
        lines.append("- Video Classification (with metadata)")
        lines.append("")
    elif dataset_type == "document":
        lines.append("## Supported Tasks")
        lines.append("")
        lines.append("- Document Understanding")
        lines.append("- Question Answering (with metadata)")
        lines.append("")

    lines.append("## Languages")
    lines.append("")
    for lang in languages:
        lines.append(f"- {lang}")
    lines.append("")
    lines.append("## Citation")
    lines.append("")
    lines.append("<!-- Leave blank. Add your citation here. -->")
    lines.append("```bibtex")
    lines.append("# TODO: Add citation")
    lines.append("```")
    lines.append("")
    lines.append("## Dataset Structure")
    lines.append("")
    if dataset_type == "audio":
        lines.append("```")
        lines.append("<dataset-root>/")
        lines.append("├── README.md          # This file")
        lines.append("├── metadata.csv        # Audio transcripts (filename|text)")
        lines.append("└── wavs/              # Audio files (.wav)")
        lines.append("    ├── utt_0001.wav")
        lines.append("    └── utt_0002.wav")
        lines.append("```")
        lines.append("")
    elif dataset_type == "image":
        lines.append("```")
        lines.append("<dataset-root>/")
        lines.append("├── README.md          # This file")
        lines.append("├── train/             # Training images")
        lines.append("│   ├── class_a/")
        lines.append("│   │   ├── img_001.jpg")
        lines.append("│   │   └── img_002.png")
        lines.append("│   └── class_b/")
        lines.append("└── test/              # Test images")
        lines.append("```")
        lines.append("")
    elif dataset_type == "video":
        lines.append("```")
        lines.append("<dataset-root>/")
        lines.append("├── README.md          # This file")
        lines.append("├── train/             # Training videos")
        lines.append("│   ├── class_a/")
        lines.append("│   │   ├── vid_001.mp4")
        lines.append("│   │   └── vid_002.mkv")
        lines.append("└── test/              # Test videos")
        lines.append("```")
        lines.append("")

    lines.append("## License")
    lines.append("")
    lines.append(f"This dataset is licensed under the {license_val} license.")
    lines.append("")
    lines.append("## Acknowledgments")
    lines.append("")
    lines.append("<!-- Add acknowledgments here. -->")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Structure validation
# ---------------------------------------------------------------------------

def validate_dataset_structure(path: Path, dataset_type: str) -> list[str]:
    """Validate dataset structure and return a list of warnings.

    Does not fail — only warns about potential issues.
    """
    warnings: list[str] = []
    path = Path(path)

    if dataset_type == "audio":
        if not (path / "wavs").exists():
            msg = "No 'wavs/' subdirectory found - audio files may not be "
            msg += "loaded correctly by HF datasets AudioFolder."
            warnings.append(msg)
        metadata = path / "metadata.csv"
        if not metadata.exists():
            warnings.append(
                "No 'metadata.csv' found - without it, audio files won't have "
                "text labels."
            )

    elif dataset_type == "image":
        has_subdirs = any(p.is_dir() for p in path.rglob("*"))
        if not has_subdirs:
            msg = "No class subdirectories found - HF ImageFolder expects "
            msg += "train/test/... structure with class labels as folder names."
            warnings.append(msg)

    elif dataset_type == "video":
        has_subdirs = any(p.is_dir() for p in path.rglob("*"))
        if not has_subdirs:
            msg = "No class subdirectories found - HF VideoFolder expects "
            msg += "train/test/... structure with class labels as folder names."
            warnings.append(msg)

    elif dataset_type == "document":
        has_meta = (path / "metadata.csv").exists() or (path / "metadata.jsonl").exists()
        if not has_meta:
            warnings.append(
                "No metadata file (metadata.csv or metadata.jsonl) found - "
                "PDF labels won't be available."
            )

    elif dataset_type == "generic":
        has_data = any(p.suffix.lower() in GENERIC_EXTENSIONS for p in path.rglob("*"))
        if not has_data:
            warnings.append(
                "No common data files (.csv, .json, .parquet, .txt) found."
            )

    return warnings


# ---------------------------------------------------------------------------
# HuggingFace API helpers
# ---------------------------------------------------------------------------

def _get_token(token_arg: str | None) -> str | None:
    """Resolve HF token from argument or environment."""
    if token_arg:
        return token_arg
    return os.environ.get("HF_TOKEN")


def _derive_repo_id(path: Path, dataset_type: str) -> str:
    """Derive a repo-id from the path name."""
    return _slugify(path.name)


# ---------------------------------------------------------------------------
# Main command
# ---------------------------------------------------------------------------

def cmd_upload(args):
    """Upload a dataset or model to HuggingFace Hub."""
    from huggingface_hub import HfApi

    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    # --- Detect or use explicit type ---
    dataset_type = args.type
    if dataset_type == "auto" or dataset_type is None:
        dataset_type = detect_dataset_type(path)
        print(f"Auto-detected dataset type: {dataset_type}")
    else:
        if dataset_type not in ("audio", "image", "video", "document", "generic", "model"):
            print(f"Error: Unknown dataset type: {dataset_type}", file=sys.stderr)
            valid = "audio, image, video, document, generic, model, auto"
            print(f"Valid types: {valid}", file=sys.stderr)
            sys.exit(1)

    # --- Determine repo_id ---
    repo_id = args.repo_id
    if not repo_id:
        repo_id = _derive_repo_id(path, dataset_type)
        print(f"No --repo-id specified. Using derived name: {repo_id}")
        if not args.yes:
            resp = input(f"Continue with repo-id '{repo_id}'? [Y/n] ").strip().lower()
            if resp in ("n", "no"):
                print("Aborted.")
                return

    # --- Parse extra metadata ---
    extra_metadata: dict | None = None
    if args.metadata:
        import yaml

        try:
            extra_metadata = yaml.safe_load(args.metadata)
            if not isinstance(extra_metadata, dict):
                print("Error: --metadata must be a YAML dictionary.", file=sys.stderr)
                sys.exit(1)
        except yaml.YAMLError as e:
            print(f"Error parsing --metadata YAML: {e}", file=sys.stderr)
            sys.exit(1)

    # --- Validate structure ---
    warnings = validate_dataset_structure(path, dataset_type)
    for w in warnings:
        print(f"Warning: {w}")

    # --- Generate dataset card ---
    readme_path = path / "README.md"
    if readme_path.exists() and not args.force:
        if args.verbose:
            print(f"README.md already exists at {readme_path}. Use --force to overwrite.")
    else:
        readme_content = generate_dataset_card(
            path,
            dataset_type,
            repo_name=repo_id.split("/")[-1] if "/" in repo_id else None,
            extra_metadata=extra_metadata,
        )
        readme_path.write_text(readme_content, encoding="utf-8")
        print(f"Generated README.md at {readme_path}")

    # --- Dry run ---
    if args.dry_run:
        print("")
        print("=== DRY RUN: would upload ===")
        print(f"  Local path:  {path}")
        print(f"  Repo ID:     {repo_id}")
        print(f"  Repo type:   {'model' if dataset_type == 'model' else 'dataset'}")
        print(f"  Private:     {args.private}")
        print(f"  Dataset type: {dataset_type}")
        print(f"  Files:       {len(list(path.rglob('*' if path.is_dir() else '')))}")
        print("=" * 40)
        print("Done! (dry-run — no files uploaded)")
        return

    # --- Confirmation ---
    if not args.yes:
        file_count = len(list(path.rglob("*"))) if path.is_dir() else 1
        resp = input(
            f"\nUpload {file_count} file(s) from '{path}' to '{repo_id}' on HuggingFace Hub? [Y/n] "
        ).strip().lower()
        if resp in ("n", "no"):
            print("Aborted.")
            return

    # --- Upload ---
    token = _get_token(args.token)
    repo_type = "model" if dataset_type == "model" else "dataset"

    api = HfApi()

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type=repo_type,
            private=args.private,
            exist_ok=True,
            token=token,
        )
        if args.verbose:
            print(f"Repository '{repo_id}' ready.")
    except Exception as e:
        print(f"Error creating/accessing repository: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo_id,
            repo_type=repo_type,
            commit_message="Upload dataset via datasety",
            token=token,
        )
    except Exception as e:
        print(f"Error uploading: {e}", file=sys.stderr)
        sys.exit(1)

    base_url = "https://huggingface.co"
    url = f"{base_url}/{repo_type}s/{repo_id}"
    print("")
    print("=" * 50)
    print(f"Successfully uploaded to: {url}")
    print("=" * 50)


def register_parser(subparsers):
    """Register CLI arguments for the upload command."""
    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload a dataset or model to HuggingFace Hub",
        description=__doc__,
    )

    upload_parser.add_argument(
        "--path", "-p",
        required=True,
        help="Path to the dataset or model directory to upload",
    )
    upload_parser.add_argument(
        "--repo-id", "-r",
        default=None,
        help="HuggingFace repo ID (e.g., 'username/my-dataset'). "
             "If omitted, derived from the directory name.",
    )
    upload_parser.add_argument(
        "--type", "-t",
        default="auto",
        choices=["audio", "image", "video", "document", "generic", "model", "auto"],
        help="Dataset or model type (default: auto-detect from structure)",
    )
    upload_parser.add_argument(
        "--private",
        action="store_true",
        help="Make the repository private",
    )
    upload_parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace API token (or set HF_TOKEN env var)",
    )
    upload_parser.add_argument(
        "--force",
        action="store_true",
        help="Force regenerate README.md if it already exists",
    )
    upload_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without actually uploading",
    )
    upload_parser.add_argument(
        "--readme-template",
        default=None,
        help="Path to a custom README template (not yet implemented)",
    )
    upload_parser.add_argument(
        "--metadata",
        default=None,
        help="Extra YAML key:value pairs for dataset card frontmatter. "
             "Example: --metadata 'license:cc-by-4.0 language: [en,fr]'",
    )
    upload_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompts",
    )
    upload_parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Print detailed progress messages",
    )

    upload_parser.set_defaults(func=cmd_upload)
