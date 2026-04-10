# Getting Started

## Installation

Install the core package (resize, align, shuffle, degrade):

```bash
pip install datasety
```

Add features as needed:

```bash
pip install datasety[caption]        # Florence-2 captioning
pip install datasety[synthetic]      # Image editing (FLUX, Qwen, SDXL, etc.)
pip install datasety[mask]           # Mask generation (SAM 3, SAM 2, CLIPSeg)
pip install datasety[filter]         # Content filtering (CLIP, NudeNet)
pip install datasety[character]      # Character dataset generation
pip install datasety[audio]          # TTS audio datasets (Whisper transcription)
pip install datasety[train]          # LoRA training (FLUX, SDXL)
pip install datasety[upload]         # Upload to HuggingFace Hub
pip install datasety[workflow]       # YAML/JSON workflow support
pip install datasety[all]            # Everything
```

Verify the installation:

```bash
datasety --version
datasety --help
```

## Quick Start

### Prepare a LoRA Training Dataset

```bash
# 1. Resize images to training resolution
datasety resize -i ./raw -o ./dataset -r 1024x1024

# 2. Generate captions with a template
datasety caption -i ./dataset -o ./dataset --template "[trigger] {{caption}}"
```

### Use a Vision API for Captions

```bash
export OPENAI_API_KEY=your-key

datasety caption -i ./dataset -o ./dataset --llm-api --model gpt-5-nano
```

### Build a TTS Audio Dataset

```bash
# From a YouTube video (flat .wav/.txt pairs by default)
datasety audio --input "https://www.youtube.com/watch?v=..." \
    --output ./tts_dataset --language en --workers 4

# With LJSpeech/Piper format (metadata.csv + wavs/)
datasety audio --input ./video.mp4 --output ./dataset --metadata

# From a directory of audio files
datasety audio --input ./recordings/ --output ./dataset \
    --normalize-numbers --workers 4

# With phoneme map filtering (drops invalid segments, requires --metadata)
datasety audio --input ./video.mp4 --output ./dataset \
    --metadata --phoneme-map /path/to/piper/config.json

# Time-slicing from a URL
datasety audio --input "https://youtube.com/watch?v=...&start=50&end=90" \
    --output ./dataset
```

### Build a Video Dataset

```bash
# From a YouTube video
datasety video --input "https://www.youtube.com/watch?v=..." \
    --output ./video_dataset --language en

# From a local video file
datasety video --input ./interview.mp4 --output ./dataset

# With frame-accurate cuts (slower, default is fast stream-copy)
datasety video --input ./video.mp4 --output ./dataset --re-encode

# Directory of clips with vocal isolation for transcription
datasety video --input ./clips/ --output ./dataset --demucs
```

### Train a LoRA Adapter (Image Fine-Tuning)

```bash
# 1. Prepare dataset
datasety resize -i ./raw -o ./dataset -r 512x512
datasety caption -i ./dataset -o ./dataset --template "photo of sks person, {{caption}}"

# 2. Train LoRA on FLUX.2-klein-base-4B (~8 GB VRAM)
datasety train --input ./dataset \
    --output ./lora/flux_lora.safetensors \
    --model black-forest-labs/FLUX.2-klein-base-4B \
    --steps 500 --lr 1e-4 --lora-rank 16
```

### Train a TTS Voice Model (Audio)

```bash
# Train a Piper TTS model (auto-installs dependencies on first run)
datasety train --input ./tts_dataset \
    --output ./voice_model \
    --backend piper \
    --model kontextox/piper-base-us \
    --steps 500

# Multi-GPU training (2x L40S, etc.)
datasety train --input ./tts_dataset \
    --output ./voice_model \
    --backend piper \
    --model kontextox/piper-base-us \
    --steps 1000 \
    --accelerator gpu \
    --devices 2

# With real-time voice testing
datasety train --input ./tts_dataset \
    --output ./voice_model \
    --backend piper \
    --model kontextox/piper-base-us \
    --test-text "Hello, this is a test of my new voice."
```

> **Note:** The `train` command has two completely separate modes — **Image (LoRA)** and **Audio (TTS)** — with different parameters. Use `--family flux/sdxl/qwen` for LoRA training, or `--backend piper` for TTS training. See the [`train`](/commands/train) docs for full parameter reference.

Supports local video/audio files, YouTube URLs, directories, and `.txt` lists. See the [`audio`](/commands/audio) docs for full options.

Supports custom providers via environment variables:

| Variable          | Description                            | Default                     |
| ----------------- | -------------------------------------- | --------------------------- |
| `OPENAI_API_KEY`  | API key                                | required for `--llm-api`    |
| `OPENAI_BASE_URL` | Custom API endpoint                    | `https://api.openai.com/v1` |
| `OPENAI_MODEL`    | Default model (when `--model` omitted) | `gpt-5-nano`                |

### Run a Workflow

Create `datasety.yaml` in your project:

```yaml
steps:
  - command: resize
    args:
      input: ./raw
      output: ./dataset
      resolution: 768x1024
      crop-position: center

  - command: caption
    args:
      input: ./dataset
      output: ./dataset
      template: "[trigger] {{caption}}"

  - command: mask
    args:
      input: ./dataset
      output: ./masks
      keywords: "face,hair"
```

Validate and execute:

```bash
datasety workflow --dry-run    # validate all steps
datasety workflow              # execute
```

### Upload to HuggingFace

Upload datasets or model adapters to HuggingFace Hub. The command auto-detects the type (audio, image, video, document, model, generic) and generates a HF-compliant README dataset card.

```bash
# Upload a TTS audio dataset
datasety upload --path ./tts_dataset --repo-id user/my-voice --type audio

# Upload a LoRA adapter
datasety upload --path ./lora_output --repo-id user/sdxl-lora --type model

# Dry-run first
datasety upload --path ./dataset --repo-id user/my-dataset --dry-run
```

Requires `HF_TOKEN` env var or `--token` argument. See the [`upload`](/commands/upload) docs for full options.

## Commands Overview

### Image Processing

| Command                        | Description                               | Extra Deps  |
| ------------------------------ | ----------------------------------------- | ----------- |
| [`resize`](/commands/resize)   | Resize and crop to target resolution      | --          |
| [`caption`](/commands/caption) | Generate captions (Florence-2 or API)     | `[caption]` |
| [`audio`](/commands/audio)     | Build TTS audio datasets from video/audio | `[audio]`   |
| [`video`](/commands/video)     | Build video datasets from video files     | `[video]`   |
| [`align`](/commands/align)     | Align control/target image pairs          | --          |
| [`mask`](/commands/mask)       | Text-prompted segmentation masks          | `[mask]`    |
| [`filter`](/commands/filter)   | Filter by content (CLIP or NudeNet)       | `[filter]`  |
| [`degrade`](/commands/degrade) | Degraded versions for upscale training    | --          |
| [`upload`](/commands/upload)   | Upload datasets/models to HuggingFace Hub | --          |

### Generation

| Command                            | Description                            | Extra Deps    |
| ---------------------------------- | -------------------------------------- | ------------- |
| [`synthetic`](/commands/synthetic) | Image editing with diffusion models    | `[synthetic]` |
| [`character`](/commands/character) | Identity-preserving character datasets | `[character]` |
| [`shuffle`](/commands/shuffle)     | Random captions from text groups       | --            |

### Automation

| Command                          | Description                         | Extra Deps   |
| -------------------------------- | ----------------------------------- | ------------ |
| [`sweep`](/commands/sweep)       | Parameter grid search for synthetic | `[workflow]` |
| [`workflow`](/commands/workflow) | Multi-step pipelines from YAML/JSON | `[workflow]` |

### Training

| Command                                    | Description                                                                   | Extra Deps |
| ------------------------------------------ | ----------------------------------------------------------------------------- | ---------- |
| [`train --family flux`](/commands/train)   | LoRA fine-tuning: FLUX.2-klein, SDXL, Qwen (images + captions → .safetensors) | `[train]`  |
| [`train --backend piper`](/commands/train) | TTS training: Piper voice models (audio dataset → .ckpt/.onnx)                | `[train]`  |

## Common Patterns

All commands that process image directories share these options:

| Option              | Description                            |
| ------------------- | -------------------------------------- |
| `--input`, `-i`     | Input directory                        |
| `--output`, `-o`    | Output directory                       |
| `--input-image`     | Single image mode (alternative to dir) |
| `--device`          | `auto`, `cpu`, `cuda`, or `mps`        |
| `--dry-run`         | Preview without making changes         |
| `--recursive`, `-R` | Search input directory recursively     |
