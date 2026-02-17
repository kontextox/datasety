"""Shared utilities for datasety commands."""

import re
import sys
from pathlib import Path


def _resolve_io_mode(args):
    """Resolve directory mode vs single-image mode.

    Returns (image_files, output_path, is_single) where output_path is a
    directory in dir mode or the explicit output file path in single mode.
    """
    has_dir = getattr(args, "input", None) and getattr(args, "output", None)
    has_single = getattr(args, "input_image", None)

    if has_single:
        input_path = Path(args.input_image)
        if not input_path.exists():
            print(f"Error: Input image '{input_path}' does not exist.")
            sys.exit(1)
        # Caption uses --output-caption; all others use --output-image
        raw = getattr(args, "output_image", None) or getattr(args, "output_caption", None)
        output_path = Path(raw) if raw else None
        return [input_path], output_path, True

    if not has_dir or not args.input or not args.output:
        print(
            "Error: Either --input/--output (directory mode) or "
            "--input-image/--output-image (single-image mode) is required."
        )
        sys.exit(1)

    return None, Path(args.output), False


def get_image_files(input_dir: Path, formats: list[str], recursive: bool = False) -> list[Path]:
    """Find all images matching the specified formats (case-insensitive)."""
    allowed = {f".{fmt.lower().strip()}" for fmt in formats}
    if recursive:
        candidates = (p for p in input_dir.rglob("*") if p.is_file())
    else:
        candidates = (p for p in input_dir.iterdir() if p.is_file())
    files = [p for p in candidates if p.suffix.lower() in allowed]
    return sorted(set(files))


def _resolve_hf_file(file_path):
    """Resolve a HuggingFace file path (GGUF or safetensors).

    Local paths are returned as-is.  HF URLs (both /blob/ and /resolve/ forms)
    are downloaded via hf_hub_download and the local cache path is returned.
    """
    if file_path is None:
        return None
    if not file_path.startswith(("https://", "http://")):
        return file_path
    # Match HF URL: https://huggingface.co/{org}/{repo}/(resolve|blob)/{rev}/{path}
    m = re.match(
        r"https?://huggingface\.co/([^/]+/[^/]+)/(?:resolve|blob)/([^/]+)/(.+)",
        file_path,
    )
    if not m:
        return file_path
    repo_id, revision, filename = m.group(1), m.group(2), m.group(3)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Error: huggingface_hub is required for downloading from HuggingFace URLs.")
        print("Run: pip install 'huggingface_hub>=0.20.0'")
        sys.exit(1)

    print(f"Downloading: {repo_id} / {filename} (rev={revision})")
    return hf_hub_download(repo_id, filename, revision=revision)


def _resolve_gguf_path(gguf_path):
    """Resolve a GGUF path. Wrapper around _resolve_hf_file for backward compat."""
    return _resolve_hf_file(gguf_path)


def resolve_device(device_arg: str) -> str:
    """Resolve device string from CLI --device argument.

    Supports 'auto', 'cpu', 'cuda', and 'mps'. Auto-detection checks CUDA
    first, then MPS (Apple Silicon), falling back to CPU.
    """
    import torch

    if device_arg == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        return "cpu"
    if device_arg == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            print("Warning: MPS not available, falling back to CPU")
            return "cpu"
    return device_arg


def get_save_kwargs(fmt: str) -> dict:
    """Return Pillow save kwargs for the given output format."""
    fmt = fmt.lower()
    if fmt in ("jpg", "jpeg"):
        return {"quality": 95, "optimize": True}
    if fmt == "webp":
        return {"quality": 95}
    if fmt == "png":
        return {"optimize": True}
    return {}
