"""Audio dataset creation pipeline for TTS training.

Extracts audio from video/URLs, isolates vocals, transcribes with VAD,
normalizes text for TTS, and outputs Piper/LJSpeech-compatible datasets.

Requires: ffmpeg on PATH. Install with: pip install datasety[audio]
"""

import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


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
    from demucs.api import Separator, save_audio

    separator = Separator(model=model, device=device)
    origin, separated = separator.separate_audio_file(str(audio_path))

    # separated is a dict: {"vocals": tensor, "other": tensor, ...}
    vocals_tensor = separated.get("vocals")
    if vocals_tensor is None:
        print(
            "Warning: No vocals stem found by Demucs. Using original audio.",
            file=sys.stderr,
        )
        return audio_path

    vocals_path = temp_dir / "vocals.wav"
    save_audio(vocals_tensor, str(vocals_path), samplerate=separator.samplerate)
    return vocals_path


def _transcribe(
    audio_path: Path,
    model_size: str,
    device: str,
    language: str | None,
    verbose: bool = False,
    vad: bool = False,
) -> list[dict]:
    """Lazy-load faster-whisper. Run transcription. Returns list of segment dicts."""
    from faster_whisper import WhisperModel

    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    kwargs = {"vad_filter": vad}
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
    try:
        from tqdm import tqdm
        pbar = tqdm(desc="Transcribing", unit="s", total=int(info.duration))
    except ImportError:
        pbar = None

    result = []
    last_end = 0.0
    for seg in segments:
        result.append({
            "start": seg.start,
            "end": seg.end,
            "text": seg.text,
        })
        if pbar is not None:
            pbar.update(int(seg.end - last_end))
            last_end = seg.end

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
    min_dur: float,
    max_dur: float,
    merge_gap: float,
    lang: str = "en",
    normalize_numbers: bool = False,
    clean_text: bool = True,
    skip_existing: bool = False,
) -> list[dict]:
    """Merge adjacent segments closer than merge_gap. Slice audio using soundfile.

    Yields chunks incrementally (files written one-by-one for real-time feedback).
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

    # If skipping existing, load existing utt numbers
    existing_nums: set[int] = set()
    if skip_existing:
        for f in output_dir.glob("utt_*.wav"):
            try:
                n = int(f.stem.split("_")[1])
                existing_nums.add(n)
            except (IndexError, ValueError):
                pass

    metadata = []
    idx = 0
    for seg in merged:
        duration = seg["end"] - seg["start"]
        if duration < min_dur or duration > max_dur:
            continue

        idx += 1
        chunk_filename = f"utt_{idx:04d}.wav"
        chunk_path = output_dir / chunk_filename

        # Skip if already exists (resume mode)
        if chunk_filename in [f.name for f in output_dir.glob("utt_*.wav")]:
            # Load text from existing CSV entry if available
            metadata.append({"filename": chunk_filename, "text": ""})
            continue

        start_sample = int(seg["start"] * samplerate)
        end_sample = int(seg["end"] * samplerate)
        chunk_data = audio_data[start_sample:end_sample]

        sf.write(str(chunk_path), chunk_data, samplerate)

        # Normalize text: clean special chars and optionally expand numbers
        clean = _normalize_text(seg["text"], lang, normalize_numbers, clean_text)

        metadata.append({
            "filename": chunk_filename,
            "text": clean,
        })
        yield metadata[-1]

    return metadata


def cmd_audio(args):
    """Main context manager: creates temp_dir, runs pipeline steps, handles errors gracefully."""
    _check_ffmpeg()

    output_dir = Path(args.output)
    wavs_dir = output_dir / "wavs"

    verbose = args.verbose
    dry_run = args.dry_run
    resume = args.resume

    if verbose:
        print(f"Input:  {args.input}")
        print(f"Output: {args.output}")
        print(f"Sample rate: {args.sample_rate}")

    # --- Check existing output ---
    existing_wavs = list(wavs_dir.glob("utt_*.wav")) if wavs_dir.exists() else []
    existing_count = len(existing_wavs)

    if existing_count > 0 and not resume and not args.overwrite:
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

    # --- Step 1: Acquire media ---
    if dry_run:
        print("=== DRY RUN: would download / copy media ===")
        print("=== DRY RUN: would extract audio ===")
        print("=== DRY RUN: would transcribe ===")
        print("=== DRY RUN: would slice and normalize ===")
        print("=" * 50)
        print("Done! (dry-run — no files written)")
        print("\n(Run without --dry-run to process)")
        return

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Determine if input is local or remote
        input_str = args.input
        if input_str.startswith(("http://", "https://", "ftp://")):
            if _is_youtube(input_str):
                print("Downloading from YouTube...")
            else:
                print("Downloading from URL...")
            media_path = _download_media(input_str, temp_path, verbose)
        else:
            media_path = Path(input_str)
            if not media_path.exists():
                print(f"Error: Input file not found: {media_path}", file=sys.stderr)
                sys.exit(1)

        # --- Step 2: Extract audio ---
        working_audio = temp_path / "working.wav"
        print(f"Extracting audio ({args.sample_rate} Hz, mono)...")
        _extract_audio(media_path, working_audio, args.sample_rate, verbose)

        # --- Step 3: Vocal isolation ---
        target_audio = working_audio
        if args.demucs:
            print(f"Isolating vocals using Demucs ({args.demucs_model})...")
            target_audio = _isolate_vocals(
                working_audio, temp_path, args.demucs_model, args.device, verbose
            )

        # --- Step 4: Transcription ---
        vad_str = " with VAD" if args.vad else ""
        print(f"Transcribing{vad_str}...")
        segments = _transcribe(
            target_audio,
            args.whisper_model,
            args.device,
            args.language,
            verbose,
            vad=args.vad,
        )
        print(f"Found {len(segments)} speech segments")

        # --- Step 5: Slice audio (incremental, files written one-by-one) ---
        print(f"Slicing audio (min={args.min_duration}s, max={args.max_duration}s)...")

        # Open CSV for appending (resume) or write (fresh)
        metadata_csv = output_dir / "metadata.csv"
        csv_mode = "a" if resume else "w"
        csv_file = open(metadata_csv, csv_mode, encoding="utf-8", newline="")
        writer = csv.writer(csv_file, delimiter="|")

        chunk_count = existing_count
        try:
            for entry in _slice_audio(
                target_audio,
                list(segments),
                wavs_dir,
                args.min_duration,
                args.max_duration,
                args.merge_gap,
                lang=args.language or "en",
                normalize_numbers=args.normalize_numbers,
                clean_text=not args.no_clean_text,
                skip_existing=resume,
            ):
                writer.writerow([entry["filename"], entry["text"]])
                chunk_count += 1
                if verbose:
                    print(f"  Created {entry['filename']} ({entry['text'][:50].strip()}...)")
                else:
                    # Brief non-verbose update every 50 chunks
                    if chunk_count % 50 == 0:
                        print(f"  ... {chunk_count} chunks written")
        finally:
            csv_file.close()

        new_count = chunk_count - existing_count
        print(f"Created {new_count} new audio chunks ({chunk_count} total)")

        if args.keep_temp:
            keep_path = Path(args.keep_temp)
            keep_path.mkdir(parents=True, exist_ok=True)
            shutil.copy(working_audio, keep_path / "working.wav")
            print(f"Temporary files kept at: {keep_path}")

    print("=" * 50)
    print(f"Done! Dataset ready at: {output_dir}")
    print(f"  - {wavs_dir}/  ({chunk_count} audio files)")
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
        help="Input source: local file path, YouTube URL, or direct media URL",
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

    audio_parser.set_defaults(func=cmd_audio)
