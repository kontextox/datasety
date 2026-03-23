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
pip install datasety[train]          # LoRA training (FLUX, SDXL)
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

# 2. Generate captions with a trigger word
datasety caption -i ./dataset -o ./dataset --trigger-word "[trigger]"
```

### Use a Vision API for Captions

```bash
export OPENAI_API_KEY=your-key

datasety caption -i ./dataset -o ./dataset --llm-api --model gpt-5-nano
```

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
      trigger-word: "[trigger]"

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

## Commands Overview

### Image Processing

| Command                        | Description                            | Extra Deps  |
| ------------------------------ | -------------------------------------- | ----------- |
| [`resize`](/commands/resize)   | Resize and crop to target resolution   | --          |
| [`caption`](/commands/caption) | Generate captions (Florence-2 or API)  | `[caption]` |
| [`align`](/commands/align)     | Align control/target image pairs       | --          |
| [`mask`](/commands/mask)       | Text-prompted segmentation masks       | `[mask]`    |
| [`filter`](/commands/filter)   | Filter by content (CLIP or NudeNet)    | `[filter]`  |
| [`inspect`](/commands/inspect) | Dataset statistics and duplicate detection | --       |
| [`degrade`](/commands/degrade) | Degraded versions for upscale training | --          |

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

| Command                    | Description                                | Extra Deps    |
| -------------------------- | ------------------------------------------ | ------------- |
| [`train`](/commands/train) | LoRA fine-tuning for FLUX.2-klein and SDXL | `[train]`     |

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
