"""Video dataset creation pipeline.

Extracts video segments based on speech transcription, producing paired
.mp4 + .txt files in a flat output directory.

Supports:
  - Single file: --input ./video.mp4
  - Directory of files: --input ./clips/ (sorted by name: 1.mp4, 2.mp4, ...)
  - Text file with lists: --input list.txt
  - YouTube/URL sources: --input "https://youtube.com/watch?v=...&start=50&end=90"

Output format (flat pairs):
  output/
  ├── 000000-000003.mp4
  ├── 000000-000003.txt
  ├── dQw4w9WgXcQ-000123-000127.mp4
  └── dQw4w9WgXcQ-000123-000127.txt

Requires: ffmpeg on PATH. Install with: pip install datasety[audio]
"""

import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from datasety.media import (
    _apply_template,
    _check_ffmpeg,
    _clean_tts_text,
    _download_media,
    _extract_audio,
    _get_source_name,
    _get_start_idx,
    _isolate_vocals,
    _load_progress,
    _make_segment_name,
    _mark_complete,
    _mark_in_progress,
    _normalize_text,
    _save_progress,
    _transcribe,
    build_media_items,
)


def _extract_video_segment(
    input_media: Path,
    output_path: Path,
    start: float,
    end: float,
    re_encode: bool = False,
    verbose: bool = False,
):
    """Extract a video segment using FFmpeg.

    Args:
        input_media: Path to source video file.
        output_path: Path for the output segment.
        start: Start time in seconds.
        end: End time in seconds.
        re_encode: If True, re-encode for frame-accurate cuts. If False, stream-copy (fast).
        verbose: Print FFmpeg output.
    """
    cmd = ["ffmpeg", "-y", "-ss", str(start), "-to", str(end), "-i", str(input_media)]

    if re_encode:
        cmd.extend(["-c:v", "libx264", "-c:a", "aac"])
    else:
        cmd.extend(["-c", "copy"])

    cmd.append(str(output_path))
    stdout = subprocess.DEVNULL if not verbose else None
    stderr = subprocess.DEVNULL if not verbose else None
    subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)


def _slice_video(
    media_path: Path,
    segments: list[dict],
    output_dir: Path,
    local_skip: int,
    min_dur: float,
    max_dur: float,
    merge_gap: float,
    re_encode: bool,
    lang: str = "en",
    normalize_numbers: bool = False,
    clean_text: bool = True,
    verbose: bool = False,
    source_name: str | None = None,
    template: str | None = None,
):
    """Slice video into segments based on transcription timestamps.

    Yields (count, seg_i, entry) tuples for each segment written.
    """
    merged = []
    for seg in segments:
        if merge_gap > 0 and merged and (seg["start"] - merged[-1]["end"]) < merge_gap:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + seg["text"]
        else:
            merged.append(seg.copy())

    count = 0
    for i, seg in enumerate(merged):
        if i < local_skip:
            continue
        duration = seg["end"] - seg["start"]
        if duration < min_dur or duration > max_dur:
            if verbose:
                reason = (
                    f"shorter than min ({duration:.2f}s < {min_dur}s)"
                    if duration < min_dur
                    else f"longer than max ({duration:.2f}s > {max_dur}s)"
                )
                print(f'    [SKIP] seg {i}: {reason} - "{seg["text"][:50]}..."')
            continue

        clean = _normalize_text(seg["text"], lang, normalize_numbers, clean_text)
        clean = _clean_tts_text(clean)

        text = _apply_template(template, clean) if template else clean

        seg_name = _make_segment_name(source_name, seg["start"], seg["end"])
        video_ext = media_path.suffix.lower()
        if video_ext not in (".mp4", ".mkv", ".avi", ".mov", ".webm"):
            video_ext = ".mp4"

        video_filename = f"{seg_name}{video_ext}"
        video_path = output_dir / video_filename

        _extract_video_segment(media_path, video_path, seg["start"], seg["end"], re_encode, verbose)

        text_path = output_dir / f"{seg_name}.txt"
        text_path.write_text(text, encoding="utf-8")

        count += 1
        entry = {"filename": video_filename, "text": text}
        yield (count, i, entry)


def _process_single_media(
    media_item: dict,
    output_dir: Path,
    args,
    temp_path: Path,
    verbose: bool,
    local_skip: int = 0,
    show_progress: bool = True,
    source_name: str | None = None,
    template: str | None = None,
):
    is_url = media_item["is_url"]
    is_youtube = media_item["is_youtube"]
    source_str = media_item["source"]
    start_time = media_item["start"]
    end_time = media_item["end"]
    name = media_item["name"]

    if is_youtube or is_url:
        if verbose:
            print(f"  Downloading from {source_str}...")
        target_media = _download_media(source_str, temp_path, verbose)
    else:
        target_media = media_item["path"]

    working_audio = temp_path / f"working_{name}.wav"
    if verbose:
        print(f"  Extracting audio from {name} for transcription...")
    _extract_audio(target_media, working_audio, 16000, verbose, start_time, end_time)

    if args.demucs:
        if verbose:
            print(f"  Isolating vocals using Demucs ({args.demucs_model})...")
        target_audio = _isolate_vocals(
            working_audio, temp_path, args.demucs_model, args.device, verbose
        )
    else:
        target_audio = working_audio

    vad_str = " with VAD" if args.vad else ""
    if verbose:
        print(f"  Transcribing{vad_str}...")
    segments = _transcribe(
        target_audio,
        args.whisper_model,
        args.device,
        args.language,
        verbose,
        vad=args.vad,
        show_progress=True,
    )
    if verbose:
        print(f"  Found {len(segments)} speech segments")

    if verbose:
        print(f"  Slicing video (min={args.min_duration}s, max={args.max_duration}s)...")

    for count, seg_i, entry in _slice_video(
        target_media,
        list(segments),
        output_dir,
        local_skip,
        args.min_duration,
        args.max_duration,
        args.merge_gap,
        args.re_encode,
        lang=args.language or "en",
        normalize_numbers=args.normalize_numbers,
        clean_text=not args.no_clean_text,
        verbose=verbose,
        source_name=source_name,
        template=template,
    ):
        yield (count, seg_i, entry)


def _deduplicate_pairs(output_dir: Path) -> None:
    """Remove consecutive duplicate text entries in flat pair mode."""
    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    media_files = sorted(
        [f for f in output_dir.iterdir() if f.is_file() and f.suffix.lower() in video_exts]
    )
    if not media_files:
        return

    import csv

    deletions = []
    prev_text = None
    for media_path in media_files:
        txt_path = media_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if text and text == prev_text:
            media_path.unlink()
            txt_path.unlink()
            deletions.append(
                {
                    "filename": media_path.name,
                    "reason": "duplicate_text",
                    "text": text,
                }
            )
        else:
            prev_text = text

    if deletions:
        log_path = output_dir / "deletions.csv"
        with open(log_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "reason", "text"])
            writer.writeheader()
            writer.writerows(deletions)
        print(f"Removed {len(deletions)} consecutive duplicate pairs")


def cmd_video(args):
    _check_ffmpeg()

    output_dir = Path(args.output)
    verbose = args.verbose
    dry_run = args.dry_run

    input_source = args.input
    if not input_source:
        print("Error: --input is required.", file=sys.stderr)
        sys.exit(1)

    media_items = build_media_items(input_source, verbose)

    if not media_items:
        print("Error: No valid media files found to process.", file=sys.stderr)
        sys.exit(1)

    is_single_source = len(media_items) == 1

    if verbose:
        print(f"Output: {args.output}")
        print("Format: flat pairs (.mp4 + .txt)")
        if args.re_encode:
            print("Encoding: re-encode (frame-accurate)")
        else:
            print("Encoding: stream-copy (fast)")

    existing_media = (
        [
            f
            for f in output_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".mp4", ".mkv", ".avi", ".mov", ".webm")
        ]
        if output_dir.exists()
        else []
    )
    existing_count = len(existing_media)

    if existing_count > 0 and not args.resume and not args.overwrite:
        print(
            f"Error: Output already has {existing_count} video files in {output_dir}/",
            file=sys.stderr,
        )
        print("Use --resume to continue, or --overwrite to start fresh.", file=sys.stderr)
        sys.exit(1)

    if args.overwrite:
        if output_dir.exists():
            video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
            for f in output_dir.iterdir():
                if f.is_file() and (f.suffix.lower() in video_exts or f.suffix.lower() == ".txt"):
                    f.unlink()
        existing_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("=== DRY RUN: would process media ===")
        for mf in media_items:
            print(f"  - {mf['source']} (start={mf['start']} end={mf['end']})")
        print("=" * 50)
        print("Done! (dry-run — no files written)")
        return

    total_chunks = existing_count

    progress = _load_progress(output_dir) if args.resume else {}
    if args.overwrite and output_dir.exists():
        progress = {}

    for file_idx, media_item in enumerate(media_items):
        with TemporaryDirectory() as temp_dir:
            filename = media_item["name"]
            file_entry = progress.get(filename, {})
            status = file_entry.get("status", "pending")
            if status == "complete":
                print(f"  Skipping {filename} (already complete)")
                continue

            start_idx = _get_start_idx(progress, filename)
            if start_idx > 0:
                print(f"  Resuming {filename} from segment {start_idx}")
                _mark_in_progress(progress, filename, start_idx)
                _save_progress(output_dir, progress)

            src_name = None if is_single_source else _get_source_name(media_item)

            print(f"[{file_idx + 1}/{len(media_items)}] Processing {filename}...")
            last_seg_i = start_idx - 1
            for count, seg_i, entry in _process_single_media(
                media_item,
                output_dir,
                args,
                Path(temp_dir),
                verbose,
                local_skip=start_idx,
                show_progress=verbose,
                source_name=src_name,
                template=getattr(args, "template", None) or None,
            ):
                total_chunks += 1
                last_seg_i = seg_i
                _mark_in_progress(progress, filename, seg_i + 1)
                _save_progress(output_dir, progress)
                if verbose:
                    print(f"  Created {entry['filename']} ({entry['text'][:50].strip()}...)")
                else:
                    if total_chunks % 50 == 0:
                        print(f"  ... {total_chunks} chunks written")
            else:
                _mark_complete(progress, filename, last_seg_i + 1)
                _save_progress(output_dir, progress)

    new_count = total_chunks - existing_count
    print(f"Created {new_count} new video segments ({total_chunks} total)")

    _deduplicate_pairs(output_dir)

    print("=" * 50)
    print(f"Done! Dataset ready at: {output_dir}")
    print(f"  - {output_dir}/  ({total_chunks} video + text pairs)")


def register_parser(subparsers):
    video_parser = subparsers.add_parser(
        "video",
        help="Build video dataset from video files (YouTube, URL, local file, or .txt list)",
        description=__doc__,
    )
    video_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input: local file, URL, directory, or .txt list. Use '?start=X&end=Y' for slicing.",
    )
    video_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for the dataset"
    )
    video_parser.add_argument(
        "--demucs",
        action="store_true",
        help="Enable Demucs vocal isolation for transcription (removes background noise/music)",
    )
    video_parser.add_argument(
        "--demucs-model", default="htdemucs", help="Demucs model name (default: htdemucs)"
    )
    video_parser.add_argument(
        "--whisper-model",
        default="base",
        help="Faster-Whisper model size: tiny, base, small, medium, large-v3",
    )
    video_parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., en, es, fr). Auto-detected if omitted.",
    )
    video_parser.add_argument(
        "--device", default="auto", help="Device for transcription: auto, cpu, cuda, mps"
    )
    video_parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable voice activity detection (VAD) to filter non-speech audio",
    )
    video_parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Minimum segment duration in seconds (default: 1.5)",
    )
    video_parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum segment duration in seconds (default: 30.0)",
    )
    video_parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="Merge segments closer than this many seconds (default: 0.0, off)",
    )
    video_parser.add_argument(
        "--re-encode",
        action="store_true",
        help="Re-encode video for frame-accurate cuts (slower, default: stream-copy)",
    )
    video_parser.add_argument(
        "--normalize-numbers",
        action="store_true",
        help="Expand digits into words (e.g., 123 -> one hundred twenty-three)",
    )
    video_parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="Disable special character stripping (keeps emojis/symbols)",
    )
    video_parser.add_argument(
        "--template",
        default="",
        help="Template for transcript text. Use {{transcript}} as placeholder. "
        "Without placeholder, text is prepended. "
        "Examples: 'sks person says: {{transcript}}', '[trigger] {{transcript}}'",
    )
    video_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run (skip existing chunks)",
    )
    video_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output directory"
    )
    video_parser.add_argument(
        "--dry-run", action="store_true", help="Print pipeline steps without executing"
    )
    video_parser.add_argument(
        "--verbose", "-V", action="store_true", help="Print detailed progress messages"
    )
    video_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel file workers (default: 1). Use >1 to process "
            "multiple files simultaneously."
        ),
    )
    video_parser.set_defaults(func=cmd_video)
