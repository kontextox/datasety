# video

Build video datasets from video files. Supports YouTube URLs, direct media URLs, local files, and directories of files. Extracts video segments based on speech transcription and outputs paired `.mp4` + `.txt` files.

## Usage

```bash
# YouTube video
datasety video --input "https://www.youtube.com/watch?v=..." --output ./dataset

# Local video file
datasety video --input ./video.mp4 --output ./dataset

# Directory of video files (sorted by name: 1.mp4, 2.mp4, ...)
datasety video --input ./clips/ --output ./dataset

# With vocal isolation for cleaner transcription
datasety video --input ./video.mp4 --output ./dataset --demucs

# Frame-accurate cuts (slower, default is stream-copy)
datasety video --input ./video.mp4 --output ./dataset --re-encode
```

## Options

| Option                | Description                                                                                               | Default     |
| --------------------- | --------------------------------------------------------------------------------------------------------- | ----------- |
| `--input`, `-i`       | Input: local file, directory, `.txt` list, YouTube/URL (append `?start=X&end=Y` for time-slicing)         | (required)  |
| `--output`, `-o`      | Output directory for the dataset                                                                          | (required)  |
| `--demucs`            | Enable Demucs vocal isolation for transcription (removes background noise/music)                          | `false`     |
| `--demucs-model`      | Demucs model name                                                                                         | `htdemucs`  |
| `--whisper-model`     | Faster-Whisper model: tiny, base, small, medium, large-v3                                                 | `base`      |
| `--language`          | Language code (e.g., en, es, fr). Auto-detected if omitted                                                | (auto)      |
| `--device`            | Device: auto, cpu, cuda, mps                                                                              | `auto`      |
| `--min-duration`      | Minimum segment duration in seconds                                                                       | `1.5`       |
| `--max-duration`      | Maximum segment duration in seconds                                                                       | `30.0`      |
| `--merge-gap`         | Merge segments closer than this many seconds                                                              | `0.0` (off) |
| `--vad`               | Enable voice activity detection (VAD) to filter non-speech                                                | `false`     |
| `--re-encode`         | Re-encode video for frame-accurate cuts (slower, default: stream-copy)                                    | `false`     |
| `--normalize-numbers` | Expand digits into words (e.g., 123 -> one hundred twenty-three)                                          | `false`     |
| `--no-clean-text`     | Disable special character stripping                                                                       | `false`     |
| `--workers`           | Number of parallel file workers (default: 1)                                                              | `1`         |
| `--resume`            | Resume a previous run (skip existing chunks)                                                              | `false`     |
| `--overwrite`         | Overwrite existing output directory                                                                       | `false`     |
| `--dry-run`           | Print pipeline steps without executing                                                                    | `false`     |
| `--verbose`, `-V`     | Print detailed progress messages                                                                          | `false`     |
| `--template`          | Template for transcript text. Use `{{transcript}}` as placeholder; without placeholder, text is prepended | (none)      |

## Output

The command creates a dataset directory with paired video and text files:

```
output/
├── 000000-000003.mp4
├── 000000-000003.txt
├── dQw4w9WgXcQ-000123-000127.mp4
└── dQw4w9WgXcQ-000123-000127.txt
```

### Naming Convention

Segment filenames use timestamp ranges in `HHMMSS` format:

| Input Source       | Segment Filename Pattern         | Example                          |
| ------------------ | -------------------------------- | -------------------------------- |
| Single local file  | `{start}-{end}.{ext}`            | `000000-000003.mp4`              |
| Directory of files | `{stem}-{start}-{end}.{ext}`     | `clip23-000000-000003.mp4`       |
| YouTube URL        | `{video_id}-{start}-{end}.{ext}` | `dQw4w9WgXcQ-000123-000127.mp4`  |
| Non-YouTube URL    | `{hash}-{start}-{end}.{ext}`     | `4a1b2c3d4e5f-000000-000005.mp4` |

For example, `012345-012347` means the segment starts at 01h 23m 45s and ends at 01h 23m 47s.

## Template System

The `--template` flag formats transcribed text using a template string:

- **With placeholder** — `{{transcript}}` is replaced with the transcribed text:
  `--template "sks person says: {{transcript}}"` → `sks person says: hello world`
- **Without placeholder** — text is prepended:
  `--template "ohwx person,"` → `ohwx person, hello world`

## Examples

### YouTube Video

Extract speech segments from a YouTube video:

```bash
datasety video \
  --input "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
  --output ./video_dataset \
  --whisper-model base \
  --language en
```

### Local Video with Vocal Isolation

For videos with background music/noise, enable Demucs to improve transcription accuracy:

```bash
datasety video \
  --input ./interview.mp4 \
  --output ./dataset \
  --demucs \
  --demucs-model htdemucs
```

### Directory of Video Files

Process a directory of video files sorted by name:

```bash
datasety video \
  --input ./videos/ \
  --output ./dataset \
  --language en
```

### Frame-Accurate Cuts

By default, segments are extracted using stream-copy (fast, no re-encoding). Use `--re-encode` for frame-accurate cuts at the cost of slower processing:

```bash
datasety video \
  --input ./video.mp4 \
  --output ./dataset \
  --re-encode
```

### High-Quality Transcription

Use a larger Whisper model for better transcription accuracy:

```bash
datasety video \
  --input ./video.mp4 \
  --output ./hq_dataset \
  --whisper-model large-v3 \
  --language en
```

### Non-English Languages

```bash
# Ukrainian
datasety video \
  --input "https://www.youtube.com/watch?v=..." \
  --output ./dataset \
  --language uk

# Spanish
datasety video \
  --input ./video.mp4 \
  --output ./dataset \
  --language es
```

### Time-Slicing from URLs

Extract only a specific segment from a video using `?start=X&end=Y` parameters:

```bash
datasety video \
  --input "https://www.youtube.com/watch?v=...&start=50&end=90" \
  --output ./dataset
```

### Text List Input

Pass a `.txt` file containing one URL/path per line:

```bash
# sources.txt
https://youtube.com/watch?v=...&start=0&end=60
https://youtube.com/watch?v=...&start=60&end=120
./local_clip.mp4
```

```bash
datasety video \
  --input sources.txt \
  --output ./dataset \
  --workers 4
```

### Dry Run

Preview what would be processed without downloading or transcribing:

```bash
datasety video \
  --input ./clips/ \
  --output ./dataset \
  --dry-run \
  --verbose
```

## Pipeline Steps

1. **Download** (if remote): Uses `yt-dlp` to download YouTube/URL media
2. **Extract audio**: FFmpeg extracts audio as mono WAV for transcription
3. **Isolate** (optional): Demucs separates vocals from background
4. **Transcribe**: Faster-Whisper identifies speech segments
5. **Slice**: Video is cut into segments matching speech timestamps, filtered by min/max duration
6. **Normalize**: Text is cleaned (special chars stripped, numbers expanded if enabled)
7. **Export**: Video chunks saved as `.mp4` (or source format), text as `.txt` sidecar files
8. **Deduplicate**: Consecutive duplicate text entries are removed

## Requirements

- **ffmpeg** must be installed and on PATH
- Install with: `pip install datasety[video]` (alias for `datasety[audio]` — both include the same dependencies)
