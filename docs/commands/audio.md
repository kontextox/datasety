# audio

Build TTS (Text-to-Speech) audio datasets from video or audio files. Supports YouTube URLs, direct media URLs, local files, and directories of files (sorted by name). Outputs Piper/LJSpeech-compatible datasets with `metadata.csv` and a `wavs/` directory.

## Usage

```bash
# YouTube video
datasety audio --input "https://www.youtube.com/watch?v=..." --output ./dataset

# Local video file
datasety audio --input ./video.mp4 --output ./dataset

# Directory of audio/video files (sorted by name: 1.mp3, 2.mp3, ...)
datasety audio --input ./clips/ --output ./dataset

# With vocal isolation (removes background noise/music)
datasety audio --input ./video.mp4 --output ./dataset --demucs

# Custom Whisper model size
datasety audio --input ./video.mp4 --output ./dataset --whisper-model large-v3 --language en
```

## Options

| Option                | Description                                                      | Default     |
| --------------------- | ---------------------------------------------------------------- | ----------- |
| `--input`, `-i`       | Input: local file, directory of audio/video files, YouTube URL, or direct media URL | (required) |
| `--output`, `-o`      | Output directory for the dataset                                 | (required)  |
| `--sample-rate`       | Output audio sample rate in Hz                                   | `22050`     |
| `--demucs`            | Enable Demucs vocal isolation (removes background noise/music)    | `false`     |
| `--demucs-model`      | Demucs model name                                                | `htdemucs`  |
| `--whisper-model`     | Faster-Whisper model: tiny, base, small, medium, large-v3       | `base`      |
| `--language`          | Language code (e.g., en, es, fr). Auto-detected if omitted      | (auto)      |
| `--device`            | Device: auto, cpu, cuda, mps                                    | `auto`      |
| `--min-duration`      | Minimum segment duration in seconds                              | `1.5`       |
| `--max-duration`      | Maximum segment duration in seconds                              | `30.0`      |
| `--merge-gap`         | Merge segments closer than this many seconds                     | `0.0` (off) |
| `--vad`               | Enable voice activity detection (VAD) to filter non-speech      | `false`     |
| `--normalize-numbers` | Expand digits into words (e.g., 123 -> one hundred twenty-three) | `false`   |
| `--no-clean-text`     | Disable special character stripping                             | `false`     |
| `--keep-temp`         | Keep temporary audio files at this path                         |             |
| `--resume`            | Resume a previous run (skip existing chunks, append to CSV)      | `false`     |
| `--overwrite`         | Overwrite existing output directory                             | `false`     |
| `--dry-run`           | Print pipeline steps without executing                          | `false`     |
| `--verbose`, `-V`     | Print detailed progress messages                                 | `false`     |

## Output

The command creates a dataset directory with the following structure:

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

## Requirements

- **ffmpeg** must be installed and on PATH
- Optional dependencies (install with `pip install datasety[audio]`):
  - `yt-dlp` - for YouTube/URL downloading
  - `demucs` - for vocal isolation
  - `faster-whisper` - for transcription
  - `soundfile` - for audio slicing
  - `num2words` - for number expansion

## Use with Piper Training

The output format is compatible with [OHF-Voice/piper1-gpl](https://github.com/OHF-Voice/piper1-gpl):

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
