"""Audio dataset creation pipeline for TTS training.

Extracts audio from video/URLs, isolates vocals, transcribes with VAD,
normalizes text for TTS, and outputs Piper/LJSpeech-compatible datasets.

Supports:
  - Single file: --input ./video.mp4
  - Directory of files: --input ./clips/ (sorted by name: 1.mp3, 2.mp3, ...)
  - Text file with lists: --input list.txt
  - YouTube/URL sources: --input "https://youtube.com/watch?v=...&start=50&end=90"

Default output format (flat pairs):
  output/
  ├── 000000-000003.wav
  ├── 000000-000003.txt
  ├── clip23-000005-000010.wav
  └── clip23-000005-000010.txt

With --metadata (LJSpeech/Piper format):
  output/
  ├── wavs/
  │   ├── utt_0001.wav
  │   └── ...
  └── metadata.csv

Requires: ffmpeg on PATH. Install with: pip install datasety[audio]
"""

import csv
import shutil
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
    _is_valid_by_phonemes,
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
    source_name: str | None = None,
    output_format: str = "ljspeech",
    output_ext: str = ".wav",
    template: str | None = None,
):
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
        clean = clean.replace("|", " ")

        if valid_chars and not _is_valid_by_phonemes(clean, valid_chars):
            if verbose:
                invalid_chars = set(
                    c
                    for c in clean
                    if c.lower() not in valid_chars and c not in valid_chars and c not in " \t\n\r"
                )
                print(f'    [SKIP] seg {i}: non-phoneme chars {invalid_chars} - "{clean[:50]}..."')
            continue

        if output_format == "ljspeech":
            idx += 1
            chunk_filename = f"utt_{idx:04d}.wav"
            chunk_path = output_dir / chunk_filename
            chunk_data = audio_data[start_sample:end_sample]
            sf.write(str(chunk_path), chunk_data, samplerate)
            text = _apply_template(template, clean) if template else clean
            metadata.append({"filename": chunk_filename, "text": text})
            yield (idx, i, metadata[-1])
        else:
            seg_name = _make_segment_name(source_name, seg["start"], seg["end"])
            chunk_filename = f"{seg_name}{output_ext}"
            chunk_path = output_dir / chunk_filename
            chunk_data = audio_data[start_sample:end_sample]
            sf.write(str(chunk_path), chunk_data, samplerate)
            text = _apply_template(template, clean) if template else clean
            text_path = output_dir / f"{seg_name}.txt"
            text_path.write_text(text, encoding="utf-8")
            metadata.append({"filename": chunk_filename, "text": text})
            yield (len(metadata), i, metadata[-1])


def _process_single_media(
    media_item: dict,
    output_dir: Path,
    args,
    temp_path: Path,
    verbose: bool,
    global_idx: int,
    local_skip: int = 0,
    show_progress: bool = True,
    valid_chars: set = None,
    source_name: str | None = None,
    output_format: str = "ljspeech",
    output_ext: str = ".wav",
    template: str | None = None,
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
        output_dir,
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
        source_name=source_name,
        output_format=output_format,
        output_ext=output_ext,
        template=template,
    ):
        yield (idx, seg_i, entry)

    if args.keep_temp:
        keep_path = Path(args.keep_temp)
        keep_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(working_audio, keep_path / f"working_{name}.wav")

    return global_idx


def _process_file_in_worker(
    item: tuple, pipeline_kwargs: dict, temp_dir: Path
) -> tuple[str, list, Path]:
    media_item, start_idx = item
    args = pipeline_kwargs["args"]
    verbose = pipeline_kwargs["verbose"]
    valid_chars = pipeline_kwargs["valid_chars"]
    output_format = pipeline_kwargs["output_format"]
    source_name = pipeline_kwargs.get("source_name")
    output_ext = pipeline_kwargs.get("output_ext", ".wav")
    template = pipeline_kwargs.get("template")

    if output_format == "ljspeech":
        temp_output_dir = temp_dir / "wavs"
    else:
        temp_output_dir = temp_dir / "output"
    temp_output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for idx, seg_i, entry in _process_single_media(
        media_item,
        temp_output_dir,
        args,
        temp_dir,
        verbose,
        global_idx=0,
        local_skip=start_idx,
        show_progress=True,
        valid_chars=valid_chars,
        source_name=source_name,
        output_format=output_format,
        output_ext=output_ext,
        template=template,
    ):
        entries.append((seg_i, entry))
    return (media_item["name"], entries, temp_output_dir)


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


def _deduplicate_pairs(output_dir: Path, output_ext: str = ".wav") -> None:
    """Remove consecutive duplicate text entries in flat pair mode."""
    media_files = sorted(
        [f for f in output_dir.iterdir() if f.suffix.lower() == output_ext.lower()]
    )
    if not media_files:
        return

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
        _write_deletion_log(output_dir, deletions)
        print(f"Removed {len(deletions)} consecutive duplicate pairs")


def cmd_audio(args):
    _check_ffmpeg()

    output_dir = Path(args.output)
    verbose = args.verbose
    dry_run = args.dry_run

    use_metadata = getattr(args, "metadata", False)
    output_format = "ljspeech" if use_metadata else "pairs"
    output_ext = ".wav"

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
        print(f"Sample rate: {args.sample_rate}")
        if use_metadata:
            print("Format: LJSpeech (metadata.csv + wavs/)")
        else:
            print("Format: flat pairs (.wav + .txt)")

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

    if output_format == "ljspeech":
        wavs_dir = output_dir / "wavs"
        existing_wavs = list(wavs_dir.glob("utt_*.wav")) if wavs_dir.exists() else []
        existing_count = len(existing_wavs)
    else:
        existing_media = (
            [f for f in output_dir.iterdir() if f.suffix.lower() == ".wav"]
            if output_dir.exists()
            else []
        )
        existing_count = len(existing_media)

    if existing_count > 0 and not args.resume and not args.overwrite:
        if output_format == "ljspeech":
            print(
                f"Error: Output already has {existing_count} audio chunks "
                f"in {output_dir / 'wavs'}/",
                file=sys.stderr,
            )
        else:
            print(
                f"Error: Output already has {existing_count} audio files in {output_dir}/",
                file=sys.stderr,
            )
        print("Use --resume to continue, or --overwrite to start fresh.", file=sys.stderr)
        sys.exit(1)

    if args.overwrite:
        if output_format == "ljspeech":
            wavs_dir = output_dir / "wavs"
            if wavs_dir.exists():
                for f in wavs_dir.glob("utt_*.wav"):
                    f.unlink()
            if (output_dir / "metadata.csv").exists():
                (output_dir / "metadata.csv").unlink()
        else:
            if output_dir.exists():
                for f in list(output_dir.glob("*.wav")) + list(output_dir.glob("*.txt")):
                    f.unlink()
        existing_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_format == "ljspeech":
        wavs_dir = output_dir / "wavs"
        wavs_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("=== DRY RUN: would process media ===")
        for mf in media_items:
            print(f"  - {mf['source']} (start={mf['start']} end={mf['end']})")
        print("=" * 50)
        print("Done! (dry-run — no files written)")
        return

    csv_file = None
    writer = None
    if output_format == "ljspeech":
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

                    src_name = None if is_single_source else _get_source_name(media_item)

                    print(f"[{file_idx + 1}/{len(media_items)}] Processing {filename}...")
                    last_seg_i = start_idx - 1

                    actual_output_dir = output_dir
                    if output_format == "ljspeech":
                        actual_output_dir = output_dir / "wavs"

                    for idx, seg_i, entry in _process_single_media(
                        media_item,
                        actual_output_dir,
                        args,
                        Path(temp_dir),
                        verbose,
                        global_idx=total_chunks,
                        local_skip=start_idx,
                        show_progress=verbose,
                        valid_chars=valid_chars,
                        source_name=src_name,
                        output_format=output_format,
                        output_ext=output_ext,
                        template=getattr(args, "template", None) or None,
                    ):
                        if output_format == "ljspeech" and writer is not None:
                            writer.writerow([entry["filename"], entry["text"]])
                            csv_file.flush()
                            total_chunks = idx
                        else:
                            total_chunks += 1
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
            if csv_file is not None:
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
                if csv_file is not None:
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

            pipeline_kwargs = {
                "args": args,
                "verbose": verbose,
                "valid_chars": valid_chars,
                "output_format": output_format,
                "output_ext": output_ext,
                "template": getattr(args, "template", None) or None,
            }

            def acquire_and_process(item, order_idx):
                import tempfile

                temp_dir = tempfile.mkdtemp()
                src_name = None if is_single_source else _get_source_name(item[0])
                pipeline_kwargs_local = {**pipeline_kwargs, "source_name": src_name}
                try:
                    return order_idx, _process_file_in_worker(
                        item, pipeline_kwargs_local, Path(temp_dir)
                    )
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
                        _, entries, temp_output_dir = worker_result
                        temp_dirs_to_cleanup.append(temp_output_dir.parent)
                        results.append((order_idx, filename, entries, temp_output_dir))
                    except Exception as e:
                        print(f"Error processing {filename}: {e}", file=sys.stderr)
                        raise

            results.sort(key=lambda x: x[0])
            total_chunks = existing_count
            for order_idx, filename, entries, temp_output_dir in results:
                if not entries:
                    _mark_complete(progress, filename, 0)
                    _save_progress(output_dir, progress)
                    continue

                entries.sort(key=lambda x: x[0])
                last_seg_i = 0
                for seg_i, entry in entries:
                    old_path = temp_output_dir / entry["filename"]
                    if old_path.exists():
                        if output_format == "ljspeech":
                            total_chunks += 1
                            new_name = f"utt_{total_chunks:04d}.wav"
                            shutil.move(str(old_path), str(output_dir / "wavs" / new_name))
                            entry["filename"] = new_name
                            if writer is not None:
                                writer.writerow([entry["filename"], entry["text"]])
                        else:
                            total_chunks += 1
                            shutil.move(str(old_path), str(output_dir / entry["filename"]))
                            old_txt = old_path.with_suffix(".txt")
                            if old_txt.exists():
                                shutil.move(
                                    str(old_txt),
                                    str(output_dir / entry["filename"].replace(output_ext, ".txt")),
                                )
                        last_seg_i = seg_i

                if csv_file is not None:
                    csv_file.flush()
                _mark_complete(progress, filename, last_seg_i + 1)
                _save_progress(output_dir, progress)

            if verbose:
                print(f"  ... {total_chunks} total chunks written")
            elif total_chunks % 50 == 0:
                print(f"  ... {total_chunks} chunks written")

        finally:
            if csv_file is not None:
                csv_file.close()
            for td in temp_dirs_to_cleanup:
                if td.exists():
                    shutil.rmtree(td, ignore_errors=True)

    new_count = total_chunks - existing_count
    print(f"Created {new_count} new audio chunks ({total_chunks} total)")

    if output_format == "ljspeech":
        if csv_file is not None:
            csv_file.close()
        _deduplicate_metadata(output_dir, output_dir / "wavs")
    else:
        _deduplicate_pairs(output_dir, output_ext)

    print("=" * 50)
    print(f"Done! Dataset ready at: {output_dir}")
    if output_format == "ljspeech":
        print(f"  - {output_dir / 'wavs'}/  ({total_chunks} audio files)")
        print(f"  - {output_dir / 'metadata.csv'}")
    else:
        print(f"  - {output_dir}/  ({total_chunks} audio + text pairs)")


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
        "--metadata",
        action="store_true",
        help="Output LJSpeech/Piper format with metadata.csv and wavs/ (default: flat pairs)",
    )
    audio_parser.add_argument(
        "--template",
        default="",
        help="Template for transcript text. Use {{transcript}} as placeholder. "
        "Without placeholder, text is prepended. "
        "Examples: 'sks person says: {{transcript}}', '[trigger] {{transcript}}'",
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
