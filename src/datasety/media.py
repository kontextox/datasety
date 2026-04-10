"""Shared media processing utilities for audio and video dataset creation.

Provides common functions for downloading, extracting, transcribing,
and segmenting media files. Used by both `datasety audio` and
`datasety video` commands to avoid duplicating functionality.
"""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

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


def _format_timestamp(seconds: float) -> str:
    """Convert seconds to zero-padded HHMMSS string.

    Examples:
        0.0      -> '000000'
        3.5      -> '000003'
        65.0     -> '000105'
        3745.0   -> '010205'
    """
    total = int(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours:02d}{minutes:02d}{secs:02d}"


def _get_source_name(media_item: dict) -> str:
    """Extract a stable source name from a media_item dict for filename prefixing.

    - YouTube URLs: extracts video ID (e.g., 'dQw4w9WgXcQ')
    - Local files: returns stem (e.g., 'clip23' from 'clip23.mp4')
    - Other URLs: returns first 12 chars of MD5 hash
    """
    source = media_item.get("source", "")
    is_yt = media_item.get("is_youtube", False)

    if is_yt:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(source)
        if "youtu.be" in parsed.netloc:
            return parsed.path.strip("/").split("/")[0]
        qs = parse_qs(parsed.query)
        if "v" in qs:
            return qs["v"][0]
        return parsed.path.strip("/").split("/")[-1]

    if media_item.get("is_url", False):
        import hashlib

        return hashlib.md5(source.encode()).hexdigest()[:12]

    return Path(media_item.get("name", source)).stem


def _make_segment_name(source_name: str | None, start: float, end: float) -> str:
    """Build a segment filename stem from source name and time range.

    Args:
        source_name: Source prefix (e.g., 'clip23', 'dQw4w9WgXcQ').
                     None or empty means single-source (no prefix).
        start: Segment start time in seconds.
        end: Segment end time in seconds.

    Returns:
        Filename stem like 'clip23-000000-000003' or '000123-000127'.
    """
    start_ts = _format_timestamp(start)
    end_ts = _format_timestamp(end)
    if source_name:
        return f"{source_name}-{start_ts}-{end_ts}"
    return f"{start_ts}-{end_ts}"


def _apply_template(template: str | None, text: str) -> str:
    """Apply a template string to text, replacing {{caption}} or {{transcript}}.

    The template can contain the placeholders ``{{caption}}`` and/or
    ``{{transcript}}`` (both are treated identically) which will be
    replaced with the provided text. If the template does not contain
    a placeholder, the text is simply prepended with the template
    (plus a space separator).

    Args:
        template: Template string, e.g. ``"photo of sks person, {{caption}}"``.
                  None or empty means no templating — return text as-is.
        text: The generated text (caption or transcript) to format.

    Returns:
        Formatted text string.

    Examples:
        >>> _apply_template("[trigger] {{caption}}", "a cat sitting")
        '[trigger] a cat sitting'
        >>> _apply_template("sks person says: {{transcript}}", "hello world")
        'sks person says: hello world'
        >>> _apply_template("photo of sks,", "a woman in red")
        'photo of sks, a woman in red'
        >>> _apply_template(None, "hello world")
        'hello world'
    """
    if not template:
        return text

    if "{{caption}}" in template or "{{transcript}}" in template:
        return template.replace("{{caption}}", text).replace("{{transcript}}", text)

    return f"{template} {text}"


def build_media_items(input_source: str, verbose: bool = False) -> list[dict]:
    """Parse input source and build a list of media_item dicts.

    Supports:
      - Single local file
      - Directory of media files
      - .txt file listing paths/URLs (one per line)
      - YouTube/URL with optional ?start=X&end=Y

    Returns list of dicts with keys:
      source, name, path, start, end, is_url, is_youtube
    """
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

    return media_items


_PROGRESS_FILE = "progress.json"


def _load_progress(output_dir: Path) -> dict:
    path = output_dir / _PROGRESS_FILE
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_progress(output_dir: Path, progress: dict) -> None:
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
