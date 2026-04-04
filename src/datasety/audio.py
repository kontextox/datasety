"""Audio dataset creation pipeline for TTS training.

Extracts audio from video/URLs, isolates vocals, transcribes with VAD,
normalizes text for TTS, and outputs Piper/LJSpeech-compatible datasets.

Supports:
  - Single file: --input ./video.mp4
  - Directory of files: --input ./clips/ (sorted by name: 1.mp3, 2.mp3, ...)
  - Text file with lists: --input list.txt
  - YouTube/URL sources: --input "https://youtube.com/watch?v=...&start=50&end=90"

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
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".webm",
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
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


def _parse_source_string(src: str) -> tuple[str, float | None, float | None]:
    """Parse custom start/end time queries from URLs or file paths."""
    start, end = None, None
    if "?" in src and ("start=" in src or "end=" in src):
        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        parsed = urlparse(src)
        qs = parse_qs(parsed.query)
        if "start" in qs:
            start = float(qs.pop("start")[0])
        if "end" in qs:
            end = float(qs.pop("end")[0])
        new_query = urlencode(qs, doseq=True)
        src = urlunparse(parsed._replace(query=new_query))
    return src, start, end


def _get_media_files(input_dir: Path) -> list[Path]:
    """Get all audio/video files from a directory, sorted by name (numeric-aware)."""
    files = []
    for p in input_dir.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)

    def _sort_key(p: Path) -> tuple[tuple[int | str, ...], str]:
        parts = re.split(r"(\d+)", p.stem)
        key = tuple(int(part) if part.isdigit() else part.lower() for part in parts)
        return (key, p.suffix.lower())

    return sorted(files, key=_sort_key)


def _download_media(source: str, temp_dir: Path, verbose: bool) -> Path:
    """Download remote media using yt-dlp. Returns path to downloaded file."""
    try:
        import yt_dlp
    except ImportError:
        print("Error: yt-dlp is required for downloading from URLs.", file=sys.stderr)
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
            candidates = [
                temp_dir / "download",
                temp_dir / "download.webm",
                temp_dir / "download.mkv",
                temp_dir / "download.mp4",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            downloaded = list(temp_dir.glob("download.*"))
            return downloaded[0] if downloaded else temp_dir / "download"
    except yt_dlp.utils.DownloadError as e:
        print(f"Error downloading media: {e}", file=sys.stderr)
        sys.exit(1)


def _extract_audio(
    input_media: Path,
    output_wav: Path,
    sample_rate: int = 22050,
    verbose: bool = False,
    start: float = None,
    end: float = None,
):
    """Extract mono audio using FFmpeg via subprocess."""
    cmd = ["ffmpeg", "-y"]
    if start is not None:
        cmd.extend(["-ss", str(start)])
    if end is not None:
        cmd.extend(["-to", str(end)])

    cmd.extend(
        [
            "-i",
            str(input_media),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            str(output_wav),
        ]
    )
    stdout = subprocess.DEVNULL if not verbose else None
    stderr = subprocess.DEVNULL if not verbose else None
    subprocess.run(cmd, stdout=stdout, stderr=stderr, check=True)


def _isolate_vocals(
    audio_path: Path, temp_dir: Path, model: str, device: str, verbose: bool = False
) -> Path:
    """Lazy-load Demucs to isolate vocals. Returns path to isolated vocals stem."""
    import julius
    import soundfile as sf
    import torch as th
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    from demucs.pretrained import get_model

    separator = get_model(model)
    separator.eval()

    wav_np, sr = sf.read(str(audio_path))
    wav_t = th.from_numpy(wav_np).float()
    if wav_t.ndim == 1:
        wav_t = wav_t.unsqueeze(0)

    wav_t = julius.resample_frac(wav_t, sr, 44100)
    wav_t = convert_audio(wav_t, 44100, 44100, 2)

    with th.no_grad():
        result = apply_model(separator, wav_t.unsqueeze(0), shifts=0, split=True, overlap=0.25)

    vocals_tensor = None
    if isinstance(result, th.Tensor) and result.ndim == 4:
        vocals_idx = separator.sources.index("vocals")
        if vocals_idx < result.shape[1]:
            vocals_tensor = result[0, vocals_idx]
    elif isinstance(result, dict):
        vocals_tensor = result.get("vocals")

    if vocals_tensor is None:
        print("Warning: No vocals stem found by Demucs. Using original audio.", file=sys.stderr)
        return audio_path

    vocals_path = temp_dir / "vocals.wav"
    vocals_np = vocals_tensor.cpu().numpy()
    if vocals_np.ndim == 1:
        sf.write(str(vocals_path), vocals_np, 44100)
    else:
        sf.write(str(vocals_path), vocals_np.T, 44100)
    return vocals_path


def _build_segment_with_word_alignment(seg) -> dict:
    if not seg.words:
        return {"start": seg.start, "end": seg.end, "text": seg.text}

    words = seg.words
    word_list = [(w.start, w.end, w.word.strip()) for w in words]

    if not word_list:
        return {"start": seg.start, "end": seg.end, "text": seg.text}

    snapped_end = word_list[-1][1]
    trimmed_text_parts = [
        wtext for start_t, end_t, wtext in word_list if end_t <= snapped_end + 0.01
    ]

    trimmed = " ".join(trimmed_text_parts).strip()
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
        lang_display = language or "auto"
        print(
            f"Transcribing with faster-whisper ({model_size}) on {device} "
            f"[language={lang_display}]..."
        )

    segments, info = model.transcribe(str(audio_path), **kwargs)

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
    text: str, lang: str, normalize_numbers: bool = False, clean_text: bool = True
) -> str:
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    if not clean_text:
        text = text.strip()
        if normalize_numbers:
            text = _expand_numbers(text, lang)
        return text

    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer

        lang_map = {
            "en": "en",
            "es": "es",
            "fr": "fr",
            "de": "de",
            "ar": "ar",
            "ru": "ru",
            "sv": "sv",
            "vi": "vi",
            "pt": "pt",
            "zh": "zh",
            "hu": "hu",
            "it": "it",
            "hy": "hy",
            "mr": "mr",
            "es_en": "es_en",
        }
        nemo_lang = lang_map.get(lang, lang)
        normalizer = Normalizer(input_case="cased", lang=nemo_lang, deterministic=True)
        text = normalizer.normalize(text, verbose=False, punct_post_process=True)
    except Exception:
        if lang == "en":
            try:
                from whisper_normalizer.english import EnglishTextNormalizer

                normalizer = EnglishTextNormalizer()
                text = normalizer(text)
            except Exception:
                text = _basic_clean_text(text)
        else:
            text = _basic_clean_text(text)

    text = text.strip()
    if normalize_numbers:
        text = _expand_numbers(text, lang)
    return text


def _basic_clean_text(text: str) -> str:
    return "".join(
        c for c in text if c.isalpha() or c.isdigit() or c.isspace() or c in ".,!?'\":;-"
    )


def _expand_numbers(text: str, lang: str) -> str:
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


def _clean_tts_text(text: str) -> str:
    text = re.sub(r"[—–]", "-", text)
    text = re.sub(r"(?<=[^\W\d_])\s*-\s*(?=[^\W\d_])", "-", text)
    text = re.sub(r"(?<=[^\W\d_])\s+'\s*(?=[^\W\d_])", "'", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(.{15,})\1{2,}", r"\1", text)
    return text.strip()


def _is_valid_by_phonemes(text: str, valid_chars: set) -> bool:
    if not valid_chars:
        return True
    allowed_extras = set(" \t\n\r")
    for char in text:
        if (
            char.lower() not in valid_chars
            and char not in valid_chars
            and char not in allowed_extras
        ):
            return False
    return True


def _slice_audio(
    audio_path: Path,
    segments: list[dict],
    output_dir: Path,
    global_idx: int,
    local_skip: int,
    min_dur: float,
    max_dur: float,
    merge_gap: float,
    lang: str = "en",
    normalize_numbers: bool = False,
    clean_text: bool = True,
    valid_chars: set = None,
    verbose: bool = False,
) -> list[dict]:
    import soundfile as sf

    audio_data, samplerate = sf.read(str(audio_path))

    merged = []
    for seg in segments:
        if merge_gap > 0 and merged and (seg["start"] - merged[-1]["end"]) < merge_gap:
            merged[-1]["end"] = seg["end"]
            merged[-1]["text"] = merged[-1]["text"].rstrip() + " " + seg["text"]
        else:
            merged.append(seg.copy())

    _RMS_WINDOW_S = 0.030
    _RMS_THRESHOLD = 0.008
    _REQUIRED_CONSECUTIVE = 3
    _MAX_LOOKAHEAD_S = 0.25

    metadata = []
    idx = global_idx
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

        start_sample = int(seg["start"] * samplerate)
        end_sample = int(seg["end"] * samplerate)

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
                    new_end_sample = e
                    break
            else:
                consecutive_loud = 0

        if i + 1 < len(merged):
            next_start = int(merged[i + 1]["start"] * samplerate)
            if new_end_sample > next_start:
                next_end_sample = int(merged[i + 1]["end"] * samplerate)
                new_end_sample = min(new_end_sample, next_end_sample - 1)
                merged[i + 1]["start"] = new_end_sample / samplerate

        end_sample = new_end_sample

        clean = _normalize_text(merged[i]["text"], lang, normalize_numbers, clean_text)
        clean = _clean_tts_text(clean)
        clean = clean.replace("|", " ")  # Prevent CSV delimiter collision

        if valid_chars and not _is_valid_by_phonemes(clean, valid_chars):
            if verbose:
                invalid_chars = set(
                    c
                    for c in clean
                    if c.lower() not in valid_chars and c not in valid_chars and c not in " \t\n\r"
                )
                print(f'    [SKIP] seg {i}: non-phoneme chars {invalid_chars} - "{clean[:50]}..."')
            continue

        idx += 1
        chunk_filename = f"utt_{idx:04d}.wav"
        chunk_path = output_dir / chunk_filename

        chunk_data = audio_data[start_sample:end_sample]
        sf.write(str(chunk_path), chunk_data, samplerate)

        metadata.append({"filename": chunk_filename, "text": clean})
        yield (idx, i, metadata[-1])


def _process_single_media(
    media_item: dict,
    wavs_dir: Path,
    args,
    temp_path: Path,
    verbose: bool,
    global_idx: int,
    local_skip: int = 0,
    show_progress: bool = True,
    valid_chars: set = None,
) -> int:
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
        print(f"  Extracting audio from {name} ({args.sample_rate} Hz, mono)...")
    _extract_audio(target_media, working_audio, args.sample_rate, verbose, start_time, end_time)

    target_audio = working_audio
    if args.demucs:
        if verbose:
            print(f"  Isolating vocals using Demucs ({args.demucs_model})...")
        target_audio = _isolate_vocals(
            working_audio, temp_path, args.demucs_model, args.device, verbose
        )

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
        print(f"  Slicing audio (min={args.min_duration}s, max={args.max_duration}s)...")

    for idx, seg_i, entry in _slice_audio(
        target_audio,
        list(segments),
        wavs_dir,
        global_idx,
        local_skip,
        args.min_duration,
        args.max_duration,
        args.merge_gap,
        lang=args.language or "en",
        normalize_numbers=args.normalize_numbers,
        clean_text=not args.no_clean_text,
        valid_chars=valid_chars,
        verbose=verbose,
    ):
        yield (idx, seg_i, entry)

    if args.keep_temp:
        keep_path = Path(args.keep_temp)
        keep_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(working_audio, keep_path / f"working_{name}.wav")

    return global_idx


_PROGRESS_FILE = "progress.json"


def _load_progress(output_dir: Path) -> dict:
    path = output_dir / _PROGRESS_FILE
    if not path.exists():
        return {}
    import json

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_progress(output_dir: Path, progress: dict) -> None:
    import json

    path = output_dir / _PROGRESS_FILE
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    tmp.rename(path)


def _mark_in_progress(progress: dict, filename: str, start_idx: int) -> None:
    progress[filename] = {"status": "in_progress", "start_idx": start_idx}


def _mark_complete(progress: dict, filename: str, chunks_written: int) -> None:
    progress[filename] = {"status": "complete", "chunks_written": chunks_written}


def _get_start_idx(progress: dict, filename: str) -> int:
    entry = progress.get(filename, {})
    return entry.get("start_idx", 0)


def _process_file_in_worker(
    item: tuple, pipeline_kwargs: dict, temp_dir: Path
) -> tuple[str, list, Path]:
    media_item, start_idx = item
    args = pipeline_kwargs["args"]
    verbose = pipeline_kwargs["verbose"]
    valid_chars = pipeline_kwargs["valid_chars"]

    temp_wavs_dir = temp_dir / "wavs"
    temp_wavs_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for idx, seg_i, entry in _process_single_media(
        media_item,
        temp_wavs_dir,
        args,
        temp_dir,
        verbose,
        global_idx=0,
        local_skip=start_idx,
        show_progress=True,
        valid_chars=valid_chars,
    ):
        entries.append((seg_i, entry))
    return (media_item["name"], entries, temp_wavs_dir)


def _write_deletion_log(output_dir: Path, deletions: list[dict]) -> None:
    log_path = output_dir / "deletions.csv"
    with open(log_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "reason", "text"])
        writer.writeheader()
        writer.writerows(deletions)
    print(f"Deletion log written to {log_path}")


def _deduplicate_metadata(output_dir: Path, wavs_dir: Path) -> None:
    """Remove consecutive duplicate text entries from metadata.csv and corresponding wav files."""
    csv_path = output_dir / "metadata.csv"
    if not csv_path.exists():
        return

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="|")
        rows = list(reader)

    if len(rows) < 2:
        return

    prev_text = None
    keep_rows: list[list[str]] = []
    deletion_log: list[dict] = []

    for row in rows:
        if len(row) < 2:
            keep_rows.append(row)
            continue
        text = row[1].strip()
        if text and text != prev_text:
            prev_text = text
            keep_rows.append(row)
        else:
            wav_path = wavs_dir / row[0]
            if wav_path.exists():
                wav_path.unlink()
            deletion_log.append(
                {
                    "filename": row[0],
                    "reason": "duplicate_text",
                    "text": text,
                }
            )

    if len(keep_rows) < len(rows):
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerows(keep_rows)
        removed = len(rows) - len(keep_rows)
        print(f"Removed {removed} consecutive duplicate entries from metadata.csv")

    if deletion_log:
        _write_deletion_log(output_dir, deletion_log)


def cmd_audio(args):
    _check_ffmpeg()

    output_dir = Path(args.output)
    wavs_dir = output_dir / "wavs"
    verbose = args.verbose
    dry_run = args.dry_run

    input_source = args.input
    if not input_source:
        print("Error: --input is required.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(input_source)
    if input_path.is_file() and input_path.suffix.lower() == ".txt":
        sources = [
            line.strip()
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if verbose:
            print(f"Loaded {len(sources)} sources from {input_path.name}")
    else:
        sources = [input_source]

    media_items = []
    for src in sources:
        src_clean, start_t, end_t = _parse_source_string(src)
        is_url = src_clean.startswith(("http://", "https://", "ftp://"))
        is_yt = _is_youtube(src_clean)

        if is_url or is_yt:
            import hashlib

            safe_name = hashlib.md5(src_clean.encode()).hexdigest()[:12] + ".url"
            media_items.append(
                {
                    "source": src_clean,
                    "name": safe_name,
                    "path": Path(safe_name),
                    "start": start_t,
                    "end": end_t,
                    "is_url": is_url,
                    "is_youtube": is_yt,
                }
            )
        else:
            p = Path(src_clean)
            if p.is_dir():
                for f in _get_media_files(p):
                    media_items.append(
                        {
                            "source": str(f),
                            "name": f.name,
                            "path": f,
                            "start": None,
                            "end": None,
                            "is_url": False,
                            "is_youtube": False,
                        }
                    )
            elif p.is_file():
                media_items.append(
                    {
                        "source": str(p),
                        "name": p.name,
                        "path": p,
                        "start": start_t,
                        "end": end_t,
                        "is_url": False,
                        "is_youtube": False,
                    }
                )
            else:
                print(f"Warning: Source not found: {src_clean}", file=sys.stderr)

    if not media_items:
        print("Error: No valid media files found to process.", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"Output: {args.output}")
        print(f"Sample rate: {args.sample_rate}")

    valid_chars = None
    if args.phoneme_map:
        pm_path = Path(args.phoneme_map)
        if pm_path.exists():
            import json

            try:
                pm_data = json.loads(pm_path.read_text(encoding="utf-8"))
                if "phoneme_id_map" in pm_data:
                    valid_chars = set(pm_data["phoneme_id_map"].keys())
                else:
                    valid_chars = set(pm_data.keys())
                if verbose:
                    print(f"Loaded phoneme map: {len(valid_chars)} valid characters.")
            except Exception as e:
                print(f"Error loading phoneme map: {e}", file=sys.stderr)
                sys.exit(1)
        else:
            print(f"Error: Phoneme map file not found: {pm_path}", file=sys.stderr)
            sys.exit(1)

    existing_wavs = list(wavs_dir.glob("utt_*.wav")) if wavs_dir.exists() else []
    existing_count = len(existing_wavs)

    if existing_count > 0 and not args.resume and not args.overwrite:
        print(
            f"Error: Output already has {existing_count} audio chunks in {wavs_dir}/",
            file=sys.stderr,
        )
        print("Use --resume to continue, or --overwrite to start fresh.", file=sys.stderr)
        sys.exit(1)

    if args.overwrite:
        if wavs_dir.exists():
            for f in wavs_dir.glob("utt_*.wav"):
                f.unlink()
        if (output_dir / "metadata.csv").exists():
            (output_dir / "metadata.csv").unlink()
        existing_count = 0

    wavs_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("=== DRY RUN: would process media ===")
        for mf in media_items:
            print(f"  - {mf['source']} (start={mf['start']} end={mf['end']})")
        print("=" * 50)
        print("Done! (dry-run — no files written)")
        return

    metadata_csv = output_dir / "metadata.csv"
    csv_mode = "a" if args.resume else "w"
    csv_file = open(metadata_csv, csv_mode, encoding="utf-8", newline="")
    writer = csv.writer(csv_file, delimiter="|")

    total_chunks = existing_count
    workers = getattr(args, "workers", 1)

    progress = _load_progress(output_dir) if args.resume else {}
    if args.overwrite and output_dir.exists():
        progress = {}

    if workers == 1:
        try:
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

                    print(f"[{file_idx + 1}/{len(media_items)}] Processing {filename}...")
                    last_seg_i = start_idx - 1
                    for idx, seg_i, entry in _process_single_media(
                        media_item,
                        wavs_dir,
                        args,
                        Path(temp_dir),
                        verbose,
                        global_idx=total_chunks,
                        local_skip=start_idx,
                        show_progress=verbose,
                        valid_chars=valid_chars,
                    ):
                        writer.writerow([entry["filename"], entry["text"]])
                        csv_file.flush()
                        total_chunks = idx
                        last_seg_i = seg_i
                        _mark_in_progress(progress, filename, seg_i + 1)
                        _save_progress(output_dir, progress)
                        if verbose:
                            print(
                                f"  Created {entry['filename']} ({entry['text'][:50].strip()}...)"
                            )
                        else:
                            if total_chunks % 50 == 0:
                                print(f"  ... {total_chunks} chunks written")
                    else:
                        _mark_complete(progress, filename, last_seg_i + 1)
                        _save_progress(output_dir, progress)
        finally:
            csv_file.close()
    else:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            pending = []
            for media_item in media_items:
                filename = media_item["name"]
                entry = progress.get(filename, {})
                status = entry.get("status", "pending")
                if status == "complete":
                    print(f"  Skipping {filename} (already complete)")
                    continue
                start_idx = _get_start_idx(progress, filename)
                pending.append((media_item, start_idx))

            if not pending:
                print("All files already processed.")
                csv_file.close()
                return

            print(f"Processing {len(pending)} files with {workers} workers...")
            has_local = any(not item[0]["is_url"] for item in pending)
            if has_local:
                if verbose:
                    print("  Pre-loading Whisper model...")
                from faster_whisper import WhisperModel

                compute_type = "float16" if args.device == "cuda" else "int8"
                WhisperModel(args.whisper_model, device=args.device, compute_type=compute_type)
                if verbose:
                    print("  Model cached.")

            pipeline_kwargs = {"args": args, "verbose": verbose, "valid_chars": valid_chars}

            def acquire_and_process(item, order_idx):
                import tempfile

                temp_dir = tempfile.mkdtemp()
                try:
                    return order_idx, _process_file_in_worker(item, pipeline_kwargs, Path(temp_dir))
                finally:
                    pass

            results = []
            temp_dirs_to_cleanup = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures_map = {
                    executor.submit(acquire_and_process, item, order_idx): item[0]["name"]
                    for order_idx, item in enumerate(pending)
                }
                for future in as_completed(futures_map):
                    filename = futures_map[future]
                    try:
                        order_idx, worker_result = future.result()
                        _, entries, temp_wavs_dir = worker_result
                        temp_dirs_to_cleanup.append(temp_wavs_dir.parent)
                        results.append((order_idx, filename, entries, temp_wavs_dir))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}", file=sys.stderr)
                        raise

            results.sort(key=lambda x: x[0])
            total_chunks = existing_count
            for order_idx, filename, entries, temp_wavs_dir in results:
                if not entries:
                    _mark_complete(progress, filename, 0)
                    _save_progress(output_dir, progress)
                    continue

                entries.sort(key=lambda x: x[0])
                last_seg_i = 0
                for seg_i, entry in entries:
                    old_path = temp_wavs_dir / entry["filename"]
                    if old_path.exists():
                        total_chunks += 1
                        new_name = f"utt_{total_chunks:04d}.wav"
                        shutil.move(str(old_path), str(wavs_dir / new_name))
                        entry["filename"] = new_name
                        writer.writerow([entry["filename"], entry["text"]])
                        last_seg_i = seg_i

                csv_file.flush()
                _mark_complete(progress, filename, last_seg_i + 1)
                _save_progress(output_dir, progress)

            if verbose:
                print(f"  ... {total_chunks} total chunks written")
            elif total_chunks % 50 == 0:
                print(f"  ... {total_chunks} chunks written")

        finally:
            csv_file.close()
            for td in temp_dirs_to_cleanup:
                if td.exists():
                    shutil.rmtree(td, ignore_errors=True)

    new_count = total_chunks - existing_count
    print(f"Created {new_count} new audio chunks ({total_chunks} total)")
    csv_file.close()
    _deduplicate_metadata(output_dir, wavs_dir)

    print("=" * 50)
    print(f"Done! Dataset ready at: {output_dir}")
    print(f"  - {wavs_dir}/  ({total_chunks} audio files)")
    print(f"  - {output_dir / 'metadata.csv'}")


def register_parser(subparsers):
    audio_parser = subparsers.add_parser(
        "audio",
        help="Build TTS audio dataset from video/audio (YouTube, URL, local file, or .txt list)",
        description=__doc__,
    )
    audio_parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Input: local file, URL, directory, or .txt list. Use '?start=X&end=Y' for slicing.",
    )
    audio_parser.add_argument(
        "--output", "-o", required=True, help="Output directory for the dataset"
    )
    audio_parser.add_argument(
        "--sample-rate",
        type=int,
        default=22050,
        help="Output audio sample rate in Hz (default: 22050)",
    )
    audio_parser.add_argument(
        "--demucs",
        action="store_true",
        help="Enable Demucs vocal isolation (removes background noise/music)",
    )
    audio_parser.add_argument(
        "--demucs-model", default="htdemucs", help="Demucs model name (default: htdemucs)"
    )
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
        "--device", default="auto", help="Device for transcription: auto, cpu, cuda, mps"
    )
    audio_parser.add_argument(
        "--vad",
        action="store_true",
        help="Enable voice activity detection (VAD) to filter non-speech audio",
    )
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
    audio_parser.add_argument(
        "--phoneme-map",
        default=None,
        help=(
            "Path to a config.json or phonemes.json file. If provided, any text "
            "segments containing characters outside this map (e.g. unexpanded "
            "numbers, foreign letters) will be automatically dropped from the "
            "dataset to ensure training safety."
        ),
    )
    audio_parser.add_argument(
        "--keep-temp", default=None, help="Keep temporary audio files at this path"
    )
    audio_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run (skip existing chunks, append to CSV)",
    )
    audio_parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output directory"
    )
    audio_parser.add_argument(
        "--dry-run", action="store_true", help="Print pipeline steps without executing"
    )
    audio_parser.add_argument(
        "--verbose", "-V", action="store_true", help="Print detailed progress messages"
    )
    audio_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Number of parallel file workers (default: 1). Use >1 to process "
            "multiple files simultaneously."
        ),
    )
    audio_parser.set_defaults(func=cmd_audio)
