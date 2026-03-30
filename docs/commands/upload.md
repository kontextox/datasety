# upload

Upload a dataset or model to [HuggingFace Hub](https://huggingface.co/). Supports audio, image, video, document, and generic datasets, plus model uploads (`.safetensors`, `.bin`, `.gguf`, `.ckpt`). Automatically generates HF-compliant README dataset cards with YAML frontmatter.

## Usage

```bash
# Audio dataset (TTS / LJSpeech structure)
datasety upload --path ./tts_dataset --repo-id user/my-tts-dataset --type audio

# Model (LoRA adapter)
datasety upload --path ./lora_output --repo-id user/sdxl-lora --type model

# Auto-detect type from structure
datasety upload --path ./my_dataset --repo-id user/my-dataset --type auto

# Dry-run — see what would be uploaded without uploading
datasety upload --path ./my_dataset --repo-id user/my-dataset --dry-run

# With extra metadata
datasety upload --path ./my_dataset --repo-id user/my-dataset \
    --metadata 'license:cc-by-4.0 language: [en,fr]'
```

## Options

| Option            | Description                                                                        | Default    |
| ----------------- | ---------------------------------------------------------------------------------- | ---------- |
| `--path`, `-p`    | Path to the dataset or model directory to upload                                   | (required) |
| `--repo-id`, `-r` | HuggingFace repo ID (e.g. `username/my-dataset`). Derived from dir name if omitted | (derived)  |
| `--type`, `-t`    | Dataset or model type                                                              | `auto`     |
| `--private`       | Make the repository private                                                        | `false`    |
| `--token`         | HuggingFace API token (or set `HF_TOKEN` env var)                                  | `HF_TOKEN` |
| `--force`         | Force regenerate README.md if it already exists                                    | `false`    |
| `--dry-run`       | Show what would be uploaded without uploading                                      | `false`    |
| `--metadata`      | Extra YAML `key: value` pairs for dataset card frontmatter                         |            |
| `--yes`, `-y`     | Skip all confirmation prompts                                                      | `false`    |
| `--verbose`, `-V` | Print detailed progress messages                                                   | `false`    |

## Supported Types

| Type       | Description                              | Auto-detection logic                          |
| ---------- | ---------------------------------------- | --------------------------------------------- |
| `audio`    | TTS / LJSpeech audio datasets            | `.wav/.mp3` files or `wavs/` + `metadata.csv` |
| `image`    | Image classification datasets            | Images with `train/`/`test/` subdirectories   |
| `video`    | Video datasets                           | `.mp4/.mkv/.mov` files                        |
| `document` | Document / PDF datasets                  | `.pdf` files                                  |
| `model`    | Model files (LoRA adapters, checkpoints) | `.safetensors/.bin/.gguf/.ckpt` files         |
| `generic`  | Tabular/text datasets                    | `.csv/.json/.parquet/.txt` files              |
| `auto`     | Auto-detect from directory structure     | Runs all detection heuristics                 |

## Auto-Detection

When `--type auto` (the default), the command inspects the directory structure to determine the dataset type:

```
Audio dataset (HF AudioFolder):
  my_dataset/
  ├── wavs/
  │   ├── utt_0001.wav
  │   └── utt_0002.wav
  └── metadata.csv

Image dataset (HF ImageFolder):
  my_dataset/
  ├── train/
  │   ├── class_a/
  │   │   └── img_001.jpg
  │   └── class_b/
  └── test/
      └── class_a/
          └── img_010.jpg

Model:
  my_lora/
  ├── adapter_model.safetensors
  └── README.md
```

## Dataset Cards

The command automatically generates a `README.md` with HF-compliant YAML frontmatter:

```yaml
---
annotations_creators:
  - no-annotation
language:
  - en
license:
  - mit
multiprocess: false
pretty_name: my-dataset
size_categories:
  - n<100
task_categories:
  - text-to-speech
dataset_modality:
  - audio
---
```

The card also includes structured sections for dataset summary, supported tasks, languages, dataset structure, citation, and license.

Use `--metadata` to add custom fields:

```bash
datasety upload --path ./dataset --repo-id user/dataset \
    --metadata 'license:cc-by-4.0 language: [en,fr] dataset_modality: [audio]'
```

## Examples

### Upload a TTS Dataset

```bash
datasety upload \
    --path ./tts_dataset \
    --repo-id your-username/my-voice-dataset \
    --type audio \
    --private
```

Assumes `tts_dataset/` has the LJSpeech structure:

```
tts_dataset/
├── wavs/
│   ├── utt_0001.wav
│   └── utt_0002.wav
└── metadata.csv
```

### Upload a LoRA Adapter

```bash
datasety upload \
    --path ./lora_output \
    --repo-id your-username/sdxl-portrait-lora \
    --type model
```

Accepts `.safetensors`, `.bin`, `.gguf`, and `.ckpt` files.

### Upload with Extra Metadata

```bash
datasety upload \
    --path ./my_dataset \
    --repo-id your-username/my-dataset \
    --metadata 'license:cc-by-4.0 language: [en,es] size_categories: [1k-2k]'
```

### Dry-Run First

Always dry-run before uploading to verify the detected type and files:

```bash
datasety upload --path ./dataset --repo-id user/dataset --dry-run --verbose
```

### Skip Confirmation

In scripts or CI, skip the yes/no prompt:

```bash
datasety upload --path ./dataset --repo-id user/dataset --yes
```

## Repository URL

After a successful upload, the repository URL is printed:

```
==================================================
Successfully uploaded to: https://huggingface.co/datasets/user/my-dataset
==================================================
```

## Environment Variables

| Variable   | Description                               |
| ---------- | ----------------------------------------- |
| `HF_TOKEN` | HuggingFace API token (used by `--token`) |

## Requirements

- `huggingface_hub>=0.20.0` (already included in `datasety` dependencies)
- `pyyaml` (for `--metadata` parsing, already included)
