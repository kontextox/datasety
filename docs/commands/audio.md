# audio

Build TTS (Text-to-Speech) audio datasets from video or audio files. Supports YouTube URLs, direct media URLs, local files, and directories of files (sorted by name). Outputs paired `.wav` + `.txt` files by default, or Piper/LJSpeech format with `--metadata`.

## Usage

```bash
# YouTube video (flat pairs output)
datasety audio --input "https://www.youtube.com/watch?v=..." --output ./dataset

# Local video file with LJSpeech/Piper format
datasety audio --input ./video.mp4 --output ./dataset --metadata

# Directory of audio/video files (sorted by name: 1.mp3, 2.mp3, ...)
datasety audio --input ./clips/ --output ./dataset

# With vocal isolation (removes background noise/music)
datasety audio --input ./video.mp4 --output ./dataset --demucs

# Custom Whisper model size
datasety audio --input ./video.mp4 --output ./dataset --whisper-model large-v3 --language en
```

## Options

| Option                | Description                                                                                                   | Default     |
| --------------------- | ------------------------------------------------------------------------------------------------------------- | ----------- |
| `--input`, `-i`       | Input: local file, directory, `.txt` list, YouTube/URL (append `?start=X&end=Y` for time-slicing)             | (required)  |
| `--output`, `-o`      | Output directory for the dataset                                                                              | (required)  |
| `--sample-rate`       | Output audio sample rate in Hz                                                                                | `22050`     |
| `--metadata`          | Output LJSpeech/Piper format with `metadata.csv` and `wavs/` directory (default: flat `.wav`/`.txt` pairs)    | `false`     |
| `--demucs`            | Enable Demucs vocal isolation (removes background noise/music)                                                | `false`     |
| `--demucs-model`      | Demucs model name                                                                                             | `htdemucs`  |
| `--whisper-model`     | Faster-Whisper model: tiny, base, small, medium, large-v3                                                     | `base`      |
| `--language`          | Language code (e.g., en, es, fr). Auto-detected if omitted                                                    | (auto)      |
| `--device`            | Device: auto, cpu, cuda, mps                                                                                  | `auto`      |
| `--min-duration`      | Minimum segment duration in seconds                                                                           | `1.5`       |
| `--max-duration`      | Maximum segment duration in seconds                                                                           | `30.0`      |
| `--merge-gap`         | Merge segments closer than this many seconds                                                                  | `0.0` (off) |
| `--vad`               | Enable voice activity detection (VAD) to filter non-speech                                                    | `false`     |
| `--normalize-numbers` | Expand digits into words (e.g., 123 -> one hundred twenty-three)                                              | `false`     |
| `--no-clean-text`     | Disable special character stripping                                                                           | `false`     |
| `--phoneme-map`       | Path to `config.json` or `phonemes.json`. Silently drops segments with unknown chars (only with `--metadata`) |             |
| `--workers`           | Number of parallel file workers (default: 1)                                                                  | `1`         |
| `--keep-temp`         | Keep temporary audio files at this path                                                                       |             |
| `--resume`            | Resume a previous run (skip existing chunks, append to CSV)                                                   | `false`     |
| `--overwrite`         | Overwrite existing output directory                                                                           | `false`     |
| `--dry-run`           | Print pipeline steps without executing                                                                        | `false`     |
| `--verbose`, `-V`     | Print detailed progress messages                                                                              | `false`     |
| `--template`          | Template for transcript text. Use `{{transcript}}` as placeholder; without placeholder, text is prepended     | (none)      |

## Output

### Default (flat pairs)

By default, the command creates paired `.wav` + `.txt` files in a flat directory:

```
output/
├── 000000-000003.wav
├── 000000-000003.txt
├── clip23-000005-000010.wav
└── clip23-000005-000010.txt
```

### With `--metadata` (LJSpeech/Piper format)

Use `--metadata` for Piper/LJSpeech-compatible output with `metadata.csv` and a `wavs/` directory:

```
output/
├── wavs/
│   ├── utt_0001.wav
│   ├── utt_0002.wav
│   └── ...
└── metadata.csv
```

The `metadata.csv` uses LJSpeech/Piper format:

```csv
utt_0001.wav|Hello world, this is a test.
utt_0002.wav|How are you doing today?
```

### Naming Convention (default mode)

Segment filenames use timestamp ranges in `HHMMSS` format:

| Input Source       | Segment Filename Pattern       | Example                          |
| ------------------ | ------------------------------ | -------------------------------- |
| Single local file  | `{start}-{end}.wav`            | `000000-000003.wav`              |
| Directory of files | `{stem}-{start}-{end}.wav`     | `clip23-000000-000003.wav`       |
| YouTube URL        | `{video_id}-{start}-{end}.wav` | `dQw4w9WgXcQ-000123-000127.wav`  |
| Non-YouTube URL    | `{hash}-{start}-{end}.wav`     | `4a1b2c3d4e5f-000000-000005.wav` |

For example, `012345-012347` means the segment starts at 01h 23m 45s and ends at 01h 23m 47s.

## Template System

The `--template` flag formats transcribed text using a template string:

- **With placeholder** — `{{transcript}}` is replaced with the transcribed text:
  `--template "sks person says: {{transcript}}"` → `sks person says: hello world`
- **Without placeholder** — text is prepended:
  `--template "ohwx person,"` → `ohwx person, hello world`

## Examples

### YouTube Video

Extract speech from a YouTube video and create a TTS dataset:

```bash
datasety audio \
  --input "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --output ./tts_dataset \
  --whisper-model base \
  --language en
```

### Local Video with Vocal Isolation

For videos with background music/noise, enable Demucs to isolate vocals:

```bash
datasety audio \
  --input ./recording.mp4 \
  --output ./clean_dataset \
  --demucs \
  --demucs-model htdemucs
```

### Directory of Audio Files

Process a directory of audio/video files sorted by name. Useful when you have pre-recorded segments like `1.mp3`, `2.mp3`, etc.:

```bash
datasety audio \
  --input ./recordings/ \
  --output ./dataset \
  --language en
```

The files are sorted numerically so `2.mp3` comes before `10.mp3`. Supported formats include MP3, WAV, FLAC, OGG, M4A, AAC, OPUS, WEBM, MP4, MKV, AVI, and MOV.

### Parallel Processing

When processing many files, use `--workers` to parallelize transcription across multiple files simultaneously:

```bash
datasety audio \
  --input ./videos/ \
  --output ./dataset \
  --workers 4
```

The transcription step is I/O-bound (waiting on model inference), so `ThreadPoolExecutor` achieves parallelism without loading multiple models into memory. Chunk indices are assigned globally after all files finish, so the `metadata.csv` is always in sorted order regardless of completion order.

### High-Quality Transcription

Use a larger Whisper model for better transcription accuracy:

```bash
datasety audio \
  --input ./video.mp4 \
  --output ./hq_dataset \
  --whisper-model large-v3 \
  --language en
```

### Number Expansion for TTS

Expand numbers to words so the TTS model knows how to pronounce them:

```bash
datasety audio \
  --input ./video.mp4 \
  --output ./dataset \
  --normalize-numbers
```

### Non-English Languages

For non-English audio, always specify the language code for accurate transcription:

```bash
# Ukrainian
datasety audio \
  --input "https://www.youtube.com/watch?v=..." \
  --output ./dataset \
  --language uk

# Spanish
datasety audio \
  --input ./video.mp4 \
  --output ./dataset \
  --language es
```

### Enabling VAD for Noisy Audio

Voice Activity Detection (VAD) filters out non-speech audio. Enable it for videos with significant background noise or music:

```bash
datasety audio \
  --input ./noisy_video.mp4 \
  --output ./dataset \
  --vad
```

VAD merges continuous speech into fewer, longer segments. Disable it (default) for clean monologue where you want fine-grained segment boundaries.

### Time-Slicing from URLs

Extract only a specific segment from a video using `?start=X&end=Y` parameters:

```bash
datasety audio \
  --input "https://www.youtube.com/watch?v=...&start=50&end=90" \
  --output ./dataset
```

Works with both YouTube URLs and local files:

```bash
datasety audio \
  --input "./video.mp4?start=10.5&end=30" \
  --output ./dataset
```

### Text List Input

Pass a `.txt` file containing one URL/path per line to process multiple sources:

```bash
# sources.txt
https://youtube.com/watch?v=...&start=0&end=60
https://youtube.com/watch?v=...&start=60&end=120
./local_clip.mp4
```

```bash
datasety audio \
  --input sources.txt \
  --output ./dataset \
  --workers 4
```

### Phoneme Map Filtering

Pass a Piper `config.json` or `phonemes.json` to silently drop any audio segments whose transcribed text contains characters (unexpanded numbers, emojis, foreign letters) not in the phoneme map. This prevents training crashes:

```bash
datasety audio \
  --input ./video.mp4 \
  --output ./dataset \
  --phoneme-map /path/to/piper/config.json
```

The script automatically detects whether you passed a full Piper config (extracts `phoneme_id_map`) or a direct phoneme map.

### Text Cleaning

The pipeline automatically cleans Whisper transcription artifacts:

- **Hallucination loops**: Removes repeated chaotic text from silent patches
- **Punctuation**: Fixes stray spaces before commas/periods, crushes double spaces
- **Hyphens**: Snaps word-connecting hyphens (`где - то` → `где-то`)
- **Apostrophes**: Connects separated apostrophes (`ім 'я` → `ім'я`)

Disable with `--no-clean-text` if you need to preserve special characters.

### Dry Run

Preview what would be processed without downloading or transcribing:

```bash
datasety audio \
  --input ./clips/ \
  --output ./dataset \
  --dry-run \
  --verbose
```

## Pipeline Steps

1. **Download** (if remote): Uses `yt-dlp` to download YouTube/URL media
2. **Extract**: FFmpeg extracts audio as mono WAV at the target sample rate
3. **Isolate** (optional): Demucs separates vocals from background
4. **Transcribe**: Faster-Whisper identifies speech segments (VAD is off by default for cleaner segmentation; use `--vad` to enable)
5. **Slice**: Audio is cut into segments matching speech timestamps, filtered by min/max duration
6. **Normalize**: Text is cleaned (special chars stripped, numbers expanded if enabled)
7. **Export**: Audio chunks saved to `wavs/`, metadata to `metadata.csv`
8. **Deduplicate**: Consecutive duplicate text entries are removed (prevents Whisper hallucinations from creating duplicate chunks)

## Requirements

- **ffmpeg** must be installed and on PATH
- Optional dependencies (install with `pip install datasety[audio]`):
  - `yt-dlp` - for YouTube/URL downloading
  - `demucs` - for vocal isolation
  - `faster-whisper` - for transcription
  - `soundfile` - for audio slicing
  - `num2words` - for number expansion

## Use with Piper Training

The `--metadata` output format is compatible with [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl):

```bash
piper-train fit \
  --data.voice_name "my_voice" \
  --data.csv_path /path/to/dataset/metadata.csv \
  --data.audio_dir /path/to/dataset/wavs/ \
  --model.sample_rate 22050 \
  --data.espeak_voice "en" \
  --data.cache_dir /path/to/cache/ \
  --data.batch_size 32
```
