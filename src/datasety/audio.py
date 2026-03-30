"""Audio dataset creation pipeline for TTS training.

Extracts audio from video/URLs, isolates vocals, transcribes with VAD,
normalizes text for TTS, and outputs Piper/LJSpeech-compatible datasets.

Supports:
  - Single file: --input ./video.mp4
  - Directory of files: --input-dir ./clips/ (sorted by name: 1.mp3, 2.mp3, ...)
  - YouTube/URL sources

Requires: ffmpeg on PATH. Install with: pip install datasety[audio]
"""

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

# Supported audio/video extensions
AUDIO_EXTENSIONS = {
    ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac",
    ".opus", ".webm", ".mp4", ".mkv", ".avi", ".mov",
}


def _check_ffmpeg():
    """Verify ffmpeg is on PATH. Exit with an actionable install message if missing."""
    if shutil.which("ffmpeg") is None:
        print("Error: ffmpeg is not installed or not on PATH.", file=sys.stderr)
        print("Install instructions:", file=sys.stderr)
        print("  macOS:     brew install ffmpeg", file=sys.stderr)
        print("  Ubuntu:    sudo apt install ffmpeg", file=sys.stderr)
        print("  Windows:   winget install ffmpeg  (or download from ffmpeg.org)", file=sys.stderr)
        sys.exit(1)


def _is_youtube(source: str) -> bool:
    """Check if source is a YouTube URL."""
    return "youtube.com" in source or "youtu.be" in source


def _get_media_files(input_dir: Path) -> list[Path]:
    """Get all audio/video files from a directory, sorted by name (numeric-aware)."""
    files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
    # Sort by name with numeric awareness: "2.mp3" comes before "10.mp3"
    def _sort_key(p: Path) -> tuple[tuple[int | str, ...], str]:
        parts = re.split(r"(\d+)", p.stem)
        key = tuple(
            int(part) if part.isdigit() else part.lower()
            for part in parts
        )
        return (key, p.suffix.lower())
    return sorted(files, key=_sort_key)


def _download_media(source: str, temp_dir: Path, verbose: bool) -> Path:
    """Download remote media using yt-dlp. Returns path to downloaded file."""
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is required for downloading from URLs.", file=sys.stderr)
        print("Install it with: pip install yt-dlp", file=sys.stderr)
        sys.exit(1)

    output_path = temp_dir / "download"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": str(output_path),
        "quiet": not verbose,
        "no_warnings": not verbose,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(source, download=True)
            # yt-dlp may change extension based on format; find the actual file
            downloaded = list(temp_dir.glob("download.*"))
            if not downloaded:
                # yt-dlp may have downloaded with no extension (e.g., "download" with webm audio)
                candidates = [
                    temp_dir / "download",
                    temp_dir / "download.webm",
                    temp_dir / "download.mkv",
                    temp_dir / "download.mp4",
                ]
                for candidate in candidates:
                    if candidate.exists():
                        return candidate
                downloaded = [temp_dir / "download"]
            return downloaded[0]
    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading media: {e}", file=sys.stderr)
        sys.exit(1)


def _extract_audio(
    input_media: Path,
    output_wav: Path,
    sample_rate: int = 22050,
    verbose: bool = False,
):
    """Extract mono audio using FFmpeg via subprocess."""
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_media),
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # 16-bit PCM
        "-ar", str(sample_rate),
        "-ac", "1",  # Mono
        str(output_wav),
    ]
    stdout = subprocess.DEVNULL if not verbose else None
    stderr = subprocess.DEVNULL if not verbose else None
    subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)


def _isolate_vocals(
    audio_path: Path,
    temp_dir: Path,
    model: str,
    device: str,
    verbose: bool = False,
) -> Path:
    """Lazy-load Demucs to isolate vocals. Returns path to isolated vocals stem."""
    import julius
    import soundfile as sf
    import torch as th
    from demucs.apply import apply_model
    from demucs.audio import convert_audio, save_audio
    from demucs.pretrained import get_model

    # Demucs 4.0: get_model returns a BagOfModels, apply_model handles separation
    separator = get_model(model)
    separator.eval()

    # Load audio using soundfile (handles mono correctly)
    # then resample to 44100 and convert to stereo for Demucs
    wav_np, sr = sf.read(str(audio_path))
    wav_t = th.from_numpy(wav_np).float()
    if wav_t.ndim == 1:
        wav_t = wav_t.unsqueeze(0)  # [1, T] for mono

    # Resample to 44100 and convert to stereo [2, T]
    wav_t = julius.resample_frac(wav_t, sr, 44100)
    wav_t = convert_audio(wav_t, 44100, 44100, 2)

    with th.no_grad():
        result = apply_model(separator, wav_t.unsqueeze(0), shifts=0, split=True, overlap=0.25)

    # Determine vocals tensor: result is [batch, sources, channels, samples]
    # For BagOfModels with htdemucs: sources = ['drums', 'bass', 'other', 'vocals']
    vocals_tensor = None
    if isinstance(result, th.Tensor) and result.ndim == 4:
        vocals_idx = separator.sources.index("vocals")
        if vocals_idx < result.shape[1]:
            vocals_tensor = result[0, vocals_idx]  # [channels, samples]
    elif isinstance(result, dict):
        vocals_tensor = result.get("vocals")
    if vocals_tensor is None:
        print(
            "Warning: No vocals stem found by Demucs. Using original audio.",
            file=sys.stderr,
        )
        return audio_path

    vocals_path = temp_dir / "vocals.wav"
    save_audio(vocals_tensor, str(vocals_path), samplerate=44100)
    return vocals_path


def _build_segment_with_word_alignment(seg) -> dict:
    """Snap segment end to word boundary and trim text to whole words only.

    Whisper's segment timestamps sometimes contain trailing text fragments.

    This function:
      1. Snaps each segment's end to the last word's actual end inside it.
      2. Trims the segment's text to only include words fully contained in it.

    Args:
        seg: a faster-whisper Segment object with `.start`, `.end`, `.text`,
             and `.words` (list of Word objects).

    Returns:
        A dict with `start`, `end`, and `text` (word-aligned and trimmed).
    """
    if not seg.words:
        # No word-level data — fall back to raw segment
        return {"start": seg.start, "end": seg.end, "text": seg.text}

    words = seg.words
    # Build a clean list: (start_time, end_time, word_text)
    word_list = []
    for w in words:
        word_list.append((w.start, w.end, w.word.strip()))

    if not word_list:
        return {"start": seg.start, "end": seg.end, "text": seg.text}

    # Snap end to the last word's actual end time
    snapped_end = word_list[-1][1]

    # Trim text to only words fully contained in this segment.
    trimmed_text_parts = []
    for start_t, end_t, wtext in word_list:
        if end_t <= snapped_end + 0.01:  # small tolerance
            trimmed_text_parts.append(wtext)

    # Join with spaces (consistent with how Whisper spaces words)
    trimmed = " ".join(trimmed_text_parts).strip()

    # Guard: if trimming broke the text badly (e.g. empty), fall back to raw
    if not trimmed:
        trimmed = seg.text.strip()

    return {"start": seg.start, "end": snapped_end, "text": trimmed}


def _transcribe(
    audio_path: Path,
    model_size: str,
    device: str,
    language: str | None,
    verbose: bool = False,
    vad: bool = False,
    show_progress: bool = True,
) -> list[dict]:
    """Lazy-load faster-whisper. Run transcription. Returns list of segment dicts."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    kwargs = {"vad_filter": vad, "word_timestamps": True}
    if vad:
        kwargs["vad_parameters"] = {"min_silence_duration_ms": 500, "threshold": 0.01}
    if language:
        kwargs["language"] = language

    if verbose:
        print(
            f"Transcribing with faster-whisper ({model_size}) on {device} "
            f"[language={language or 'auto'}]..."
        )

    segments, info = model.transcribe(str(audio_path), **kwargs)

    # Show progress bar during transcription (faster-whisper yields segments incrementally)
    # Only show in the main thread — tqdm is not thread-safe in worker threads.
    pbar = None
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(desc="Transcribing", unit="s", total=int(info.duration))
        except ImportError:
            pbar = None

    result = []
    last_end = 0.0
    for seg in segments:
        seg_dict = _build_segment_with_word_alignment(seg)
        result.append(seg_dict)
        if pbar is not None:
            pbar.update(int(seg_dict["end"] - last_end))
            last_end = seg_dict["end"]

    if pbar is not None:
        pbar.close()

    if verbose:
        print(f"  Done: {len(result)} segments from {info.duration:.0f}s of audio")

    return result


def _normalize_text(
    text: str,
    lang: str,
    normalize_numbers: bool = False,
    clean_text: bool = True,
) -> str:
    """Normalize text for TTS: strip non-pronounceable chars, optionally expand numbers.

    Uses nemo-text-processing (NVIDIA NeMo) when available for multi-language support.
    Falls back to whisper-normalizer for English, then basic cleaning.
    """
    # Strip control characters first (always)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    if not clean_text:
        text = text.strip()
        if normalize_numbers:
            text = _expand_numbers(text, lang)
        return text

    # Try nemo-text-processing first (supports 15+ languages)
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer

        # Map lang codes to NeMo-supported codes
        lang_map = {
            "en": "en", "es": "es", "fr": "fr", "de": "de", "ar": "ar",
            "ru": "ru", "sv": "sv", "vi": "vi", "pt": "pt", "zh": "zh",
            "hu": "hu", "it": "it", "hy": "hy", "mr": "mr",
            "es_en": "es_en",
        }
        nemo_lang = lang_map.get(lang, lang)
        normalizer = Normalizer(input_case="cased", lang=nemo_lang, deterministic=True)
        text = normalizer.normalize(text, verbose=False, punct_post_process=True)
    except Exception:
        # Fallback to whisper-normalizer for English
        if lang == "en":
            try:
                from whisper_normalizer.english import EnglishTextNormalizer
                normalizer = EnglishTextNormalizer()
                text = normalizer(text)
            except Exception:
                text = _basic_clean_text(text)
        else:
            # For non-English without NeMo: basic cleaning preserving all scripts
            text = _basic_clean_text(text)

    text = text.strip()

    if normalize_numbers:
        text = _expand_numbers(text, lang)

    return text


def _basic_clean_text(text: str) -> str:
    """Basic text cleaning preserving Unicode letters, numbers, punctuation, and spaces."""
    return "".join(
        c for c in text
        if c.isalpha() or c.isdigit() or c.isspace() or c in ".,!?'\":;-"
    )


def _expand_numbers(text: str, lang: str) -> str:
    """Expand digit sequences to words using num2words."""
    try:
        from num2words import num2words
    except ImportError:
        return text

    def replace_number(match):
        number = match.group(0)
        try:
            return num2words(int(number), lang=lang)
        except Exception:
            return number

    return re.sub(r"\d+", replace_number, text)


def _slice_audio(
    audio_path: Path,
    segments: list[dict],
    output_dir: Path,
    start_idx: int,
    min_dur: float,
    max_dur: float,
    merge_gap: float,
    lang: str = "en",
    normalize_numbers: bool = False,
    clean_text: bool = True,
) -> list[dict]:
    """Merge adjacent segments closer than merge_gap. Slice audio using soundfile.

    Yields (idx, entry) tuples where idx is the global chunk index (starting from start_idx).
    """
    import soundfile as sf

    audio_data, samplerate = sf.read(str(audio_path))

    # Merge segments that are close together
    merged = []
    for seg in segments:
        if merge_gap > 0 and merged and (seg["start"] - merged[-1]["end"]) < merge_gap:
            # Extend previous segment
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + seg["text"]
        else:
            merged.append(seg.copy())

    # Skip already-processed segments on resume (advances both audio slicing AND text)
    if start_idx > 0:
        merged = merged[start_idx:]

    # Silence threshold for trailing audio detection
    # Whisper sometimes marks a word as ending mid-phoneme due to a brief
    # silence gap, then the same word continues. We detect this by scanning
    # forward from each segment's end: if sustained speech (>2 consecutive
    # loud windows) is found within 250ms, extend the slice.
    _RMS_WINDOW_S = 0.030  # 30ms window for RMS measurement
    _RMS_THRESHOLD = 0.008  # RMS above this = speech
    _REQUIRED_CONSECUTIVE = 3  # 3 consecutive loud windows = real speech
    _MAX_LOOKAHEAD_S = 0.25  # max ms to look ahead

    metadata = []
    idx = start_idx
    for i, seg in enumerate(merged):
        duration = seg["end"] - seg["start"]
        if duration < min_dur or duration > max_dur:
            continue

        idx += 1
        chunk_filename = f"utt_{idx:04d}.wav"
        chunk_path = output_dir / chunk_filename

        start_sample = int(seg["start"] * samplerate)
        end_sample = int(seg["end"] * samplerate)

        # Scan forward from end_sample: if sustained speech is found within
        # _MAX_LOOKAHEAD_S, extend the slice to capture the trailing phoneme.
        rms_win = int(_RMS_WINDOW_S * samplerate)
        max_lookahead = min(int(_MAX_LOOKAHEAD_S * samplerate), len(audio_data) - end_sample)
        consecutive_loud = 0
        new_end_sample = end_sample

        for offset in range(0, max_lookahead, rms_win):
            s = end_sample + offset
            e = min(s + rms_win, len(audio_data))
            window = audio_data[s:e]
            rms = (window**2).mean() ** 0.5
            if rms > _RMS_THRESHOLD:
                consecutive_loud += 1
                if consecutive_loud >= _REQUIRED_CONSECUTIVE:
                    # Found sustained speech: extend to capture trailing audio
                    new_end_sample = e
                    break
            else:
                consecutive_loud = 0

        # --- BOUNDARY RESOLUTION FIX ---
        # Instead of clamping the current segment's end backward (which splits the trailing
        # phoneme in half and forces the orphaned audio into the next segment),
        # we let the current segment finish the word, and push the NEXT segment's
        # start forward so it doesn't pick up the trailing audio.
        if i + 1 < len(merged):
            next_start = int(merged[i + 1]["start"] * samplerate)

            # If our phoneme extension bled into the next segment's timeline
            if new_end_sample > next_start:
                next_end_sample = int(merged[i + 1]["end"] * samplerate)

                # Ensure we don't push past the next segment's actual end
                new_end_sample = min(new_end_sample, next_end_sample - 1)

                # Push the next segment's start time forward to properly hand off the audio
                merged[i + 1]["start"] = new_end_sample / samplerate

        end_sample = new_end_sample
        chunk_data = audio_data[start_sample:end_sample]

        sf.write(str(chunk_path), chunk_data, samplerate)

        # Normalize text: clean special chars and optionally expand numbers
        clean = _normalize_text(merged[i]["text"], lang, normalize_numbers, clean_text)

        metadata.append({
            "filename": chunk_filename,
            "text": clean,
        })
        yield (idx, metadata[-1])

    return metadata


def _process_single_media(
    media_path: Path,
    wavs_dir: Path,
    args,
    temp_path: Path,
    verbose: bool,
    start_idx: int,
    base_offset: int = 0,
    show_progress: bool = True,
) -> int:
    """Process a single media file through the audio pipeline.

    Yields (idx, entry) tuples for each chunk. Returns the final idx after processing.
    """
    # --- Step 2: Extract audio ---
    working_audio = temp_path / f"working_{media_path.stem}.wav"
    if verbose:
        print(f"  Extracting audio from {media_path.name} ({args.sample_rate} Hz, mono)...")
    _extract_audio(media_path, working_audio, args.sample_rate, verbose)

    # --- Step 3: Vocal isolation ---
    target_audio = working_audio
    if args.demucs:
        if verbose:
            print(f"  Isolating vocals using Demucs ({args.demucs_model})...")
        target_audio = _isolate_vocals(
            working_audio, temp_path, args.demucs_model, args.device, verbose
        )

    # --- Step 4: Transcription ---
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

    # --- Step 5: Slice audio (incremental, files written one-by-one) ---
    if verbose:
        print(
            f"  Slicing audio (min={args.min_duration}s, max={args.max_duration}s)..."
        )

    for (idx, entry) in _slice_audio(
        target_audio,
        list(segments),
        wavs_dir,
        start_idx,
        args.min_duration,
        args.max_duration,
        args.merge_gap,
        lang=args.language or "en",
        normalize_numbers=args.normalize_numbers,
        clean_text=not args.no_clean_text,
    ):
        yield (idx, entry)

    if args.keep_temp:
        keep_path = Path(args.keep_temp)
        keep_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(working_audio, keep_path / f"working_{media_path.stem}.wav")

    return start_idx


# ---------------------------------------------------------------------------
# Per-file progress tracking
# ---------------------------------------------------------------------------

_PROGRESS_FILE = "progress.json"


def _load_progress(output_dir: Path) -> dict:
    """Load progress file. Returns empty dict if none exists."""
    path = output_dir / _PROGRESS_FILE
    if not path.exists():
        return {}
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_progress(output_dir: Path, progress: dict) -> None:
    """Atomically write progress file."""
    import json
    path = output_dir / _PROGRESS_FILE
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    tmp.rename(path)


def _mark_in_progress(progress: dict, filename: str, start_idx: int) -> None:
    """Mark a file as in-progress."""
    progress[filename] = {"status": "in_progress", "start_idx": start_idx}


def _mark_complete(progress: dict, filename: str, chunks_written: int) -> None:
    """Mark a file as complete."""
    progress[filename] = {"status": "complete", "chunks_written": chunks_written}


def _get_start_idx(progress: dict, filename: str) -> int:
    """Get the start_idx for a file from progress."""
    entry = progress.get(filename, {})
    return entry.get("start_idx", 0)


# ---------------------------------------------------------------------------
# Thread-pool friendly file processor
# ---------------------------------------------------------------------------

def _process_file_in_worker(
    media_path: Path,
    pipeline_kwargs: dict,
    temp_dir: Path,
    start_idx: int,
) -> tuple[str, list, Path]:
    """Process one media file inside a worker thread.

    Writes audio chunks to a private temp subdirectory to avoid collisions.
    Returns (filename, [(idx, entry), ...], temp_wavs_dir) where temp_wavs_dir
    is the directory containing the written wav files.
    """
    args = pipeline_kwargs["args"]
    verbose = pipeline_kwargs["verbose"]
    temp_wavs_dir = temp_dir / "wavs"
    temp_wavs_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for (idx, entry) in _process_single_media(
        media_path,
        temp_wavs_dir,  # private dir — no filename collisions across workers
        args,
        temp_dir,
        verbose,
        start_idx,
        0,  # base_offset: worker uses local indices; main thread re-indexes
        show_progress=True,  # workers show tqdm bars so user sees per-file progress
    ):
        entries.append((idx, entry))
    return (media_path.name, entries, temp_wavs_dir)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def cmd_audio(args):
    """Main context manager: creates temp_dir, runs pipeline steps, handles errors gracefully."""
    _check_ffmpeg()

    output_dir = Path(args.output)
    wavs_dir = output_dir / "wavs"

    verbose = args.verbose
    dry_run = args.dry_run

    # --- Check for valid input ---
    input_source = args.input
    if not input_source:
        print("Error: --input is required.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(input_source)
    is_dir = input_path.is_dir()
    is_url = input_source.startswith(("http://", "https://", "ftp://"))
    is_youtube = _is_youtube(input_source)

    if verbose:
        print(f"Output: {args.output}")
        print(f"Sample rate: {args.sample_rate}")

    # --- Check existing output ---
    existing_wavs = list(wavs_dir.glob("utt_*.wav")) if wavs_dir.exists() else []
    existing_count = len(existing_wavs)

    if existing_count > 0 and not args.resume and not args.overwrite:
        print(
            f"Error: Output already has {existing_count} audio chunks in {wavs_dir}/",
            file=sys.stderr,
        )
        print(
            "Use --resume to continue, or --overwrite to start fresh.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.overwrite:
        # Remove existing output for fresh start
        if wavs_dir.exists():
            for f in wavs_dir.glob("utt_*.wav"):
                f.unlink()
        if (output_dir / "metadata.csv").exists():
            (output_dir / "metadata.csv").unlink()
        existing_count = 0

    wavs_dir.mkdir(parents=True, exist_ok=True)

    # --- Determine media sources ---
    media_files: list[Path] = []
    if is_dir:
        media_files = _get_media_files(input_path)
        if not media_files:
            print(
                f"Error: No audio/video files found in {input_path}.",
                file=sys.stderr,
            )
            sys.exit(1)
        if verbose:
            print(f"Input dir: {input_path} ({len(media_files)} files)")
    elif is_url:
        media_files = [Path("__url__")]
    elif is_youtube:
        media_files = [Path("__youtube__")]
    else:
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            sys.exit(1)
        media_files = [input_path]

    if verbose:
        for mf in media_files:
            print(f"Input:  {mf}")

    # --- Dry run ---
    if dry_run:
        print("=== DRY RUN: would process media ===")
        for mf in media_files:
            print(f"  - {mf}")
        print("=== DRY RUN: would extract audio ===")
        print("=== DRY RUN: would transcribe ===")
        print("=== DRY RUN: would slice and normalize ===")
        print("=" * 50)
        print("Done! (dry-run — no files written)")
        print("\n(Run without --dry-run to process)")
        return

    # --- Open CSV for appending (resume) or write (fresh) ---
    metadata_csv = output_dir / "metadata.csv"
    csv_mode = "a" if args.resume else "w"
    csv_file = open(metadata_csv, csv_mode, encoding="utf-8", newline="")
    writer = csv.writer(csv_file, delimiter="|")

    total_chunks = existing_count
    workers = getattr(args, "workers", 1)

    # --- Load progress for resume ---
    progress = _load_progress(output_dir) if args.resume else {}
    if args.overwrite and output_dir.exists():
        progress = {}

    # --- Sequential mode (workers == 1) ---
    if workers == 1:
        try:
            for idx, media_path in enumerate(media_files):
                is_youtube = str(media_path) == "__youtube__"
                is_url = str(media_path) == "__url__"

                with TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    # --- Acquire media ---
                    if is_youtube:
                        print(f"[{idx + 1}/{len(media_files)}] Downloading from YouTube...")
                        resolved = _download_media(input_source, temp_path, verbose)
                    elif is_url:
                        print(f"[{idx + 1}/{len(media_files)}] Downloading from URL...")
                        resolved = _download_media(input_source, temp_path, verbose)
                    else:
                        print(
                            f"[{idx + 1}/{len(media_files)}] Processing {media_path.name}..."
                        )
                        resolved = media_path

                    # --- Check progress (skip complete, resume in_progress) ---
                    filename = media_path.name
                    entry = progress.get(filename, {})
                    status = entry.get("status", "pending")
                    if status == "complete":
                        print(f"  Skipping {media_path.name} (already complete)")
                        continue

                    start_idx = _get_start_idx(progress, filename)
                    if start_idx > 0:
                        print(f"  Resuming {media_path.name} from chunk {start_idx}")
                        _mark_in_progress(progress, filename, start_idx)
                        _save_progress(output_dir, progress)

                    # Process the media file
                    for (idx, entry) in _process_single_media(
                        resolved,
                        wavs_dir,
                        args,
                        temp_path,
                        verbose,
                        total_chunks,
                        show_progress=verbose,
                    ):
                        writer.writerow([entry["filename"], entry["text"]])
                        total_chunks = idx
                        # Update progress after each chunk
                        _mark_in_progress(progress, media_path.name, idx)
                        _save_progress(output_dir, progress)
                        if verbose:
                            print(
                                f"  Created {entry['filename']} "
                                f"({entry['text'][:50].strip()}...)"
                            )
                        else:
                            if total_chunks % 50 == 0:
                                print(f"  ... {total_chunks} chunks written")
                    else:
                        # File finished normally (for-else: no break)
                        _mark_complete(progress, media_path.name, total_chunks)
                        _save_progress(output_dir, progress)

        finally:
            csv_file.close()

    # --- Parallel mode (workers > 1) ---
    else:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Build list of files to process (skip complete).
            # Each file gets a private temp/wavs/ subdir, so no filename collisions.
            pending = []
            for media_path in media_files:
                filename = media_path.name
                entry = progress.get(filename, {})
                status = entry.get("status", "pending")
                if status == "complete":
                    print(f"  Skipping {filename} (already complete)")
                    continue
                start_idx = _get_start_idx(progress, filename)
                pending.append((media_path, start_idx))

            if not pending:
                print("All files already processed.")
                csv_file.close()
                return

            print(f"Processing {len(pending)} files with {workers} workers...")

            # Pre-load Whisper model in main thread so it's cached in workers.
            # Downloading from HuggingFace inside a ThreadPool worker causes
            # tqdm thread-safety issues (AttributeError on disabled_tqdm._lock).
            if not is_youtube and not is_url:
                if verbose:
                    print("  Pre-loading Whisper model...")
                from faster_whisper import WhisperModel
                compute_type = "float16" if args.device == "cuda" else "int8"
                WhisperModel(args.whisper_model, device=args.device, compute_type=compute_type)
                if verbose:
                    print("  Model cached.")

            pipeline_kwargs = {
                "args": args,
                "verbose": verbose,
            }

            def acquire_and_process(item):
                """Download (if needed) and process one media file in a worker."""
                media_path, start_idx = item
                is_youtube = str(media_path) == "__youtube__"
                is_url = str(media_path) == "__url__"

                # Create temp dir manually (not with statement) so it lives until
                # the main thread has moved the wav files out of it.
                import tempfile
                temp_dir = tempfile.mkdtemp()
                temp_path = Path(temp_dir)

                try:
                    if is_youtube:
                        resolved = _download_media(str(args.input), temp_path, verbose)
                    elif is_url:
                        resolved = _download_media(str(args.input), temp_path, verbose)
                    else:
                        resolved = media_path

                    return _process_file_in_worker(
                        resolved,
                        pipeline_kwargs,
                        temp_path,
                        start_idx,
                    )
                finally:
                    # Keep temp_dir alive until main thread is done with it.
                    # Cleanup happens in the outer finally block via temp_dirs_to_cleanup.
                    pass

            # Submit all tasks and collect results
            all_entries = []
            temp_dirs_to_cleanup = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(acquire_and_process, item): item[0].name
                    for item in pending
                }
                for future in as_completed(futures):
                    filename = futures[future]
                    try:
                        _, entries, temp_wavs_dir = future.result()
                        temp_dirs_to_cleanup.append(temp_wavs_dir.parent)
                        all_entries.extend(entries)
                    except Exception as e:
                        print(f"Error processing {filename}: {e}", file=sys.stderr)
                        raise

            # Sort all chunks globally by their local chunk index.
            all_entries.sort(key=lambda x: x[0])

            # Re-index and move wavs from temp dirs to shared wavs/
            for global_idx, (_, entry) in enumerate(all_entries, start=existing_count + 1):
                old_path = None
                for td in temp_dirs_to_cleanup:
                    candidate = td / "wavs" / entry["filename"]
                    if candidate.exists():
                        old_path = candidate
                        break
                if old_path:
                    new_name = f"utt_{global_idx:04d}.wav"
                    old_path.rename(wavs_dir / new_name)
                    entry["filename"] = new_name

            # Write all entries to CSV in sorted order
            total_chunks = existing_count
            for (_, entry) in all_entries:
                writer.writerow([entry["filename"], entry["text"]])
                total_chunks += 1

            # Mark all as complete
            for media_path, _ in pending:
                _mark_complete(progress, media_path.name, total_chunks)
            _save_progress(output_dir, progress)

            if verbose:
                print(f"  ... {len(all_entries)} chunks written")
            elif total_chunks % 50 == 0:
                print(f"  ... {total_chunks} chunks written")

        finally:
            csv_file.close()

    new_count = total_chunks - existing_count
    print(f"Created {new_count} new audio chunks ({total_chunks} total)")

    print("=" * 50)
    print(f"Done! Dataset ready at: {output_dir}")
    print(f"  - {wavs_dir}/  ({total_chunks} audio files)")
    print(f"  - {output_dir / 'metadata.csv'}")


def register_parser(subparsers):
    """Register CLI arguments for the audio command."""
    audio_parser = subparsers.add_parser(
        "audio",
        help="Build TTS audio dataset from video/audio (YouTube, URL, or local file)",
        description=__doc__,
    )

    # Input / Output
    audio_parser.add_argument(
        "--input", "-i", required=True,
        help="Input: local file, directory of audio/video files, YouTube URL, or direct media URL",
    )
    audio_parser.add_argument(
        "--output", "-o", required=True,
        help="Output directory for the dataset",
    )

    # Processing
    audio_parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Output audio sample rate in Hz (default: 22050)",
    )

    # Vocal Isolation
    audio_parser.add_argument(
        "--demucs",
        action="store_true",
        help="Enable Demucs vocal isolation (removes background noise/music)",
    )
    audio_parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs model name (default: htdemucs)",
    )

    # Transcription
    audio_parser.add_argument(
        "--whisper-model",
        default="base",
        help="Faster-Whisper model size: tiny, base, small, medium, large-v3",
    )
    audio_parser.add_argument(
        "--language",
        default=None,
        help="Language code (e.g., en, es, fr). Auto-detected if omitted.",
    )
    audio_parser.add_argument(
        "--device",
        default="auto",
        help="Device for transcription: auto, cpu, cuda, mps",
    )
    audio_parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable voice activity detection (VAD) to filter non-speech audio",
    )

    # Segmentation
    audio_parser.add_argument(
        "--min-duration",
        type=float,
        default=1.5,
        help="Minimum segment duration in seconds (default: 1.5)",
    )
    audio_parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum segment duration in seconds (default: 30.0)",
    )
    audio_parser.add_argument(
        "--merge-gap",
        type=float,
        default=0.0,
        help="Merge segments closer than this many seconds (default: 0.0, off)",
    )

    # Text Normalization
    audio_parser.add_argument(
        "--normalize-numbers",
        action="store_true",
        help="Expand digits into words (e.g., 123 -> one hundred twenty-three)",
    )
    audio_parser.add_argument(
        "--no-clean-text",
        action="store_true",
        help="Disable special character stripping (keeps emojis/symbols)",
    )

    # General
    audio_parser.add_argument(
        "--keep-temp",
        default=None,
        help="Keep temporary audio files at this path",
    )
    audio_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run (skip existing chunks, append to CSV)",
    )
    audio_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory",
    )
    audio_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print pipeline steps without executing",
    )
    audio_parser.add_argument(
        "--verbose", "-V",
        action="store_true",
        help="Print detailed progress messages",
    )
    audio_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel file workers (default: 1). "
             "Use >1 to process multiple files simultaneously.",
    )

    audio_parser.set_defaults(func=cmd_audio)
