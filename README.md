# datasety

<img align="right" src="https://raw.githubusercontent.com/kontextox/datasety/refs/heads/main/docs/public/mascot.png" alt="CLI tool for dataset preparation" width="120" />

[![PyPI](https://img.shields.io/pypi/v/datasety)](https://pypi.org/project/datasety/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

CLI tool for dataset preparation — resize, caption, align, shuffle, synthetic editing, masking, degradation, character generation, LoRA training, and multi-step workflows.

[Full documentation →](https://kontextox.github.io/datasety/)

## Installation

```bash
pip install datasety                 # core (resize, align, shuffle, degrade)
pip install datasety[caption]        # + Florence-2 captioning
pip install datasety[synthetic]      # + image editing (FLUX, Qwen, SDXL)
pip install datasety[mask]           # + segmentation masks (SAM 3, CLIPSeg)
pip install datasety[filter]         # + content filtering (CLIP, NudeNet)
pip install datasety[character]      # + character dataset generation
pip install datasety[workflow]       # + YAML workflow support
pip install datasety[train]          # + LoRA training (FLUX, SDXL)
pip install datasety[all]            # everything
```

---

## Commands

### `resize` — Resize & Crop Images

Batch resize images to exact dimensions with configurable crop positions.

<!-- screenshot: resize -->

```bash
datasety resize --input ./raw --output ./resized --resolution 768x1024 --crop-position top
```

<details>
<summary>Options</summary>

| Option                  | Description                                    | Default             |
| ----------------------- | ---------------------------------------------- | ------------------- |
| `--input`, `-i`         | Input directory                                | required\*          |
| `--output`, `-o`        | Output directory                               | required\*          |
| `--input-image`         | Single input image (alternative to dir mode)   |                     |
| `--output-image`        | Single output image (use with `--input-image`) |                     |
| `--resolution`, `-r`    | Target resolution (`WIDTHxHEIGHT`)             |                     |
| `--megapixel`           | Target megapixel count (e.g., 0.5, 1.0)        |                     |
| `--aspect-ratio`        | Aspect ratio `W:H` (e.g., 1:1, 16:9)           |                     |
| `--crop-position`       | `top`, `center`, `bottom`, `left`, `right`     | `center`            |
| `--input-format`        | Comma-separated input formats                  | `jpg,jpeg,png,webp` |
| `--output-format`       | `jpg`, `png`, `webp`                           | `jpg`               |
| `--output-name-numbers` | Rename output files to 1.jpg, 2.jpg, ...       | off                 |
| `--upscale`             | Upscale images smaller than target             | off                 |
| `--min-resolution`      | Skip images below this size (e.g., `256x256`)  |                     |
| `--workers`             | Parallel workers for processing                | `1`                 |
| `--recursive`, `-R`     | Search input directory recursively             | off                 |
| `--progress`            | Show tqdm progress bar                         | off                 |
| `--dry-run`             | Preview without modifying files                | off                 |

</details>

```bash
# Single image
datasety resize --input-image photo.jpg --output-image resized.jpg -r 512x512

# Batch with sequential numbering
datasety resize -i ./photos -o ./dataset -r 1024x1024 --output-name-numbers --crop-position top
```

[Full documentation →](https://kontextox.github.io/datasety/commands/resize)

---

### `caption` — Generate Image Captions

Generate captions using Florence-2 (local) or OpenAI-compatible vision APIs.

<!-- screenshot: caption -->

```bash
datasety caption --input ./images --output ./captions --trigger-word "[trigger]"
```

<details>
<summary>Options</summary>

| Option               | Description                                 | Default                   |
| -------------------- | ------------------------------------------- | ------------------------- |
| `--input`, `-i`      | Input directory                             | required\*                |
| `--output`, `-o`     | Output directory for .txt files             | required\*                |
| `--input-image`      | Single input image                          |                           |
| `--output-caption`   | Single output .txt path                     |                           |
| `--device`           | `auto`, `cpu`, `cuda`, `mps`                | `auto`                    |
| `--trigger-word`     | Text to prepend to each caption             |                           |
| `--prompt`           | Florence-2 task prompt                      | `<MORE_DETAILED_CAPTION>` |
| `--model`            | HF model name or API model ID               |                           |
| `--num-beams`        | Beam search width (1 = greedy)              | `3`                       |
| `--florence-2-base`  | Use Florence-2-base (0.23B, faster)         | default                   |
| `--florence-2-large` | Use Florence-2-large (0.77B, more accurate) |                           |
| `--llm-api`          | Use OpenAI-compatible vision API            |                           |
| `--max-tokens`       | Max response tokens (API mode)              | `300`                     |
| `--temperature`      | Temperature (API mode)                      | `0.3`                     |
| `--skip-existing`    | Skip images that already have a .txt file   | off                       |
| `--append`           | Append text to existing captions            |                           |
| `--prepend`          | Prepend text to existing captions           |                           |
| `--recursive`, `-R`  | Search input directory recursively          | off                       |
| `--progress`         | Show tqdm progress bar                      | off                       |
| `--dry-run`          | Preview without processing                  | off                       |

</details>

```bash
# Florence-2 with trigger word
datasety caption -i ./dataset -o ./dataset --trigger-word "photo of sks person," --device cuda

# OpenAI vision API (supports OPENAI_MODEL env var)
datasety caption -i ./images -o ./captions --llm-api --model gpt-5-nano
```

[Full documentation →](https://kontextox.github.io/datasety/commands/caption)

---

### `align` — Align Control/Target Pairs

Match dimensions, enforce multiples of 32, and unify formats for control/target training pairs. Includes a built-in web server for visual comparison with a compare slider, caption editing, and pair management.

<!-- screenshot: align -->

```bash
datasety align --target ./target --control ./control --dry-run
```

<details>
<summary>Options</summary>

| Option              | Description                              | Default       |
| ------------------- | ---------------------------------------- | ------------- |
| `--target`, `-t`    | Target images directory                  | required      |
| `--control`, `-c`   | Control images directory                 | required      |
| `--multiple-of`     | Align dimensions to this multiple        | `32`          |
| `--output-format`   | Convert all images: `jpg`, `png`, `webp` | keep original |
| `--recursive`, `-R` | Search input directories recursively     | off           |
| `--dry-run`         | Preview changes without modifying files  | off           |
| `--server`          | Start web server for visual comparison   | off           |
| `--port`            | Port for the comparison web server       | `8787`        |

</details>

```bash
# Preview, then apply
datasety align -t ./target -c ./control --dry-run
datasety align -t ./target -c ./control --output-format jpg

# Visual comparison web UI
datasety align -t ./target -c ./control --server
datasety align -t ./target -c ./control --server --port 9000
```

<img src="https://raw.githubusercontent.com/kontextox/datasety/refs/heads/main/docs/public/demo.gif" alt="The web server provides a compare slider to visually inspect control/target pairs">

The web server provides a compare slider to visually inspect control/target pairs, edit `.txt` caption files, and delete pairs — with keyboard navigation and mobile support.

[Full documentation →](https://kontextox.github.io/datasety/commands/align)

---

### `shuffle` — Random Caption Generation

Generate random captions by picking one variant from each text group.

<!-- screenshot: shuffle -->

```bash
datasety shuffle -i ./images -o ./captions \
    --group "A photo of a person.|Portrait of someone." \
    --group "Remove the hat.|Take off the hat."
```

<details>
<summary>Options</summary>

| Option                | Description                                | Default  |
| --------------------- | ------------------------------------------ | -------- |
| `--input`, `-i`       | Input directory containing images          | required |
| `--output`, `-o`      | Output directory for .txt files            | required |
| `--group`, `-g`       | Inline `\|`-separated, `.txt` file, or URL | required |
| `--separator`         | Separator between groups                   | `" "`    |
| `--seed`              | Random seed for reproducibility            |          |
| `--dry-run`           | Preview captions without writing           | off      |
| `--show-distribution` | Show caption distribution after generation | off      |

</details>

```bash
# Mix file, URL, and inline sources
datasety shuffle -i ./images -o ./captions \
    --group subjects.txt \
    --group "ending A|ending B" \
    --seed 42 --show-distribution
```

[Full documentation →](https://kontextox.github.io/datasety/commands/shuffle)

---

### `synthetic` — Synthetic Image Editing

Generate synthetic variations using image editing models (FLUX.2-klein FP8, Qwen, SDXL, LongCat, HunyuanImage). The default model `FLUX.2-klein-4b-fp8` requires no HuggingFace token and fits in ~5 GB VRAM.

<!-- screenshot: synthetic -->

```bash
datasety synthetic --input ./images --output ./synthetic --prompt "add a winter hat" --steps 4
```

<details>
<summary>Options</summary>

| Option              | Description                              | Default                                 |
| ------------------- | ---------------------------------------- | --------------------------------------- |
| `--input`, `-i`     | Input directory                          | required\*                              |
| `--output`, `-o`    | Output directory                         | required\*                              |
| `--input-image`     | Single input image                       |                                         |
| `--output-image`    | Single output image                      |                                         |
| `--prompt`, `-p`    | Edit instruction                         | required                                |
| `--model`           | Model (auto-detects family or API model) | `black-forest-labs/FLUX.2-klein-4b-fp8` |
| `--image-api`       | Use OpenAI-compatible API for generation | off                                     |
| `--weights`         | Fine-tuned weights file                  |                                         |
| `--lora`            | LoRA adapter (repeatable, `:WEIGHT`)     |                                         |
| `--device`          | `auto`, `cpu`, `cuda`, `mps`             | `auto`                                  |
| `--cpu-offload`     | Force CPU offload                        | auto                                    |
| `--steps`           | Inference steps                          | `4`                                     |
| `--cfg-scale`       | Guidance scale                           | `2.5`                                   |
| `--true-cfg-scale`  | True CFG (Qwen only)                     | `4.0`                                   |
| `--negative-prompt` | Negative prompt                          | `" "`                                   |
| `--num-images`      | Images per input                         | `1`                                     |
| `--seed`            | Random seed                              |                                         |
| `--gguf`            | GGUF path/URL for quantized loading      |                                         |
| `--strength`        | Img2img strength (SDXL/FLUX.2, 0.0-1.0)  | `0.7`                                   |
| `--recursive`, `-R` | Search input directory recursively       | off                                     |
| `--output-format`   | `png`, `jpg`, `webp`                     | `png`                                   |
| `--skip-existing`   | Skip images with existing output         | off                                     |
| `--batch-size`      | Flush GPU memory every N images          | `0` (off)                               |
| `--progress`        | Show tqdm progress bar                   | off                                     |
| `--dry-run`         | Preview without loading models           | off                                     |

</details>

```bash
# Single image edit
datasety synthetic --input-image photo.jpg --output-image edited.png \
    --prompt "add sunglasses" --steps 4

# Cloud API (no GPU needed)
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  datasety synthetic -i ./images -o ./synthetic \
  --prompt "add a winter hat" --image-api --model black-forest-labs/flux.2-klein-4b

# Qwen with LoRA
datasety synthetic -i ./dataset -o ./synthetic \
    --model "Qwen/Qwen-Image-Edit-2511" \
    --lora "adapter.safetensors:0.8" \
    --prompt "add a red scarf" --steps 40
```

[Full documentation →](https://kontextox.github.io/datasety/commands/synthetic)

---

### `mask` — Text-Prompted Segmentation Masks

Generate binary masks from images using text keywords. Supports SAM 3, SAM 2, and CLIPSeg.

<!-- screenshot: mask -->

```bash
datasety mask --input ./dataset --output ./masks --keywords "face,hair" --device cuda
```

<details>
<summary>Options</summary>

| Option              | Description                        | Default    |
| ------------------- | ---------------------------------- | ---------- |
| `--input`, `-i`     | Input directory                    | required\* |
| `--output`, `-o`    | Output directory for masks         | required\* |
| `--input-image`     | Single input image                 |            |
| `--output-image`    | Single output mask                 |            |
| `--keywords`, `-k`  | Comma-separated keywords           | required   |
| `--model`           | `sam3`, `sam2`, `clipseg`          | `sam3`     |
| `--device`          | `auto`, `cpu`, `cuda`, `mps`       | `auto`     |
| `--threshold`       | Confidence threshold (0.0-1.0)     | `0.3`      |
| `--padding`         | Pixels to expand mask (dilation)   | `0`        |
| `--blur`            | Gaussian blur radius for edges     | `0`        |
| `--invert`          | Invert mask colors                 | off        |
| `--naming`          | `folder` or `suffix` (`_mask`)     | `folder`   |
| `--output-format`   | `png`, `jpg`, `webp`               | `png`      |
| `--skip-existing`   | Skip images with existing masks    | off        |
| `--dry-run`         | Preview detections without saving  | off        |
| `--recursive`, `-R` | Search input directory recursively | off        |
| `--progress`        | Show tqdm progress bar             | off        |

</details>

```bash
# CLIPSeg (lightweight, no extra deps)
datasety mask -i ./dataset -o ./masks -k "face" --model clipseg --threshold 0.5

# SAM 2 with mask refinement
datasety mask -i ./dataset -o ./masks -k "hat,glasses" --model sam2 --padding 5 --blur 3
```

[Full documentation →](https://kontextox.github.io/datasety/commands/mask)

---

### `filter` — Filter Dataset by Content

Filter, curate, or clean datasets based on image content. Use CLIP for arbitrary text queries or NudeNet for NSFW label detection.

<!-- screenshot: filter -->

```bash
datasety filter --input ./dataset --output ./rejected --query "leg,male face" --action move
```

<details>
<summary>Options</summary>

| Option                 | Description                                             | Default  |
| ---------------------- | ------------------------------------------------------- | -------- |
| `--input`, `-i`        | Input directory                                         | required |
| `--output`, `-o`       | Output directory for matched/rejected images            |          |
| `--query`, `-q`        | Comma-separated text queries (CLIP)                     |          |
| `--labels`, `-l`       | Comma-separated NudeNet labels                          |          |
| `--model`              | `clip`, `nudenet`                                       | `clip`   |
| `--action`             | `move`, `copy`, `delete`, `keep`                        | `move`   |
| `--threshold`          | Confidence threshold (0.0-1.0)                          | `0.5`    |
| `--device`             | `auto`, `cpu`, `cuda`, `mps`                            | `auto`   |
| `--confirm`            | Required for destructive actions (`delete`, `keep`)     | off      |
| `--preserve-structure` | Keep subfolder hierarchy in output (with `--recursive`) | off      |
| `--invert`             | Invert match logic (act on non-matches)                 | off      |
| `--log`                | Write CSV log of all decisions to this path             |          |
| `--dry-run`            | Preview detections without modifying files              | off      |
| `--recursive`, `-R`    | Search input directory recursively                      | off      |
| `--progress`           | Show tqdm progress bar                                  | off      |

</details>

```bash
# Move images containing legs or male faces to a reject folder
datasety filter -i ./dataset -o ./rejected --query "leg,male face" --action move

# Delete NSFW images using NudeNet labels
datasety filter -i ./dataset --labels "FEMALE_BREAST_EXPOSED,MALE_GENITALIA_EXPOSED" \
    --action delete --model nudenet --threshold 0.6 --confirm

# Keep only images with "hat and socks", move the rest out
datasety filter -i ./dataset -o ./rejected --query "hat and socks" --action keep

# Dry-run to preview what would be filtered
datasety filter -i ./dataset --query "blurry,low quality" --action delete --dry-run -R

# Write a decision log for review
datasety filter -i ./dataset -o ./rejected --query "outdoor" --action copy --log filter_log.csv
```

[Full documentation →](https://kontextox.github.io/datasety/commands/filter)

---

### `inspect` — Dataset Statistics

Scan a dataset directory and report image count, resolution distribution, format breakdown, file sizes, caption coverage, and optionally detect duplicate images via perceptual hashing.

<!-- screenshot: inspect -->

```bash
datasety inspect --input ./dataset --duplicates
```

<details>
<summary>Options</summary>

| Option              | Description                               | Default  |
| ------------------- | ----------------------------------------- | -------- |
| `--input`, `-i`     | Input directory                           | required |
| `--duplicates`      | Detect duplicate/near-duplicate images    | off      |
| `--json`            | Export report as JSON to this path        |          |
| `--csv`             | Export per-image data as CSV to this path |          |
| `--recursive`, `-R` | Search input directory recursively        | off      |

</details>

```bash
# Full report with duplicate detection
datasety inspect -i ./dataset --duplicates

# Export report to JSON
datasety inspect -i ./dataset --json report.json

# Export per-image data to CSV
datasety inspect -i ./dataset --csv images.csv -R
```

[Full documentation →](https://kontextox.github.io/datasety/commands/inspect)

---

### `degrade` — Image Degradation

Create degraded versions of images for upscale/enhance training. Pure Pillow, no extra dependencies.

<!-- screenshot: degrade -->

```bash
datasety degrade --input ./originals --output ./dataset --type random --intensity-range 0.2-0.8 --paired
```

<details>
<summary>Options</summary>

| Option              | Description                           | Default    |
| ------------------- | ------------------------------------- | ---------- |
| `--input`, `-i`     | Input directory                       | required\* |
| `--output`, `-o`    | Output directory                      | required\* |
| `--input-image`     | Single input image                    |            |
| `--output-image`    | Single output image                   |            |
| `--type`, `-t`      | Degradation type(s), repeatable       | `random`   |
| `--intensity`       | Global intensity (0.0-1.0)            | `0.5`      |
| `--intensity-range` | Random range `MIN-MAX`                |            |
| `--chain`           | Apply multiple types sequentially     | off        |
| `--num-variants`    | Variants per input image              | `1`        |
| `--paired`          | Create `control/` + `target/` subdirs | off        |
| `--seed`            | Random seed                           |            |
| `--output-format`   | `png`, `jpg`, `webp`                  | `png`      |
| `--skip-existing`   | Skip images with existing output      | off        |
| `--workers`         | Parallel workers for processing       | `1`        |
| `--progress`        | Show tqdm progress bar                | off        |
| `--dry-run`         | Preview without writing files         | off        |

**Degradation types:** `lowres`, `oversharpen`, `noise`, `blur`, `jpeg`, `motion-blur`, `pixelate`, `color-bands`, `upscale-sim`, `random`

</details>

```bash
# Chain specific degradations for paired output
datasety degrade -i ./images -o ./dataset --type jpeg --type noise --chain --paired --seed 42

# Multiple random variants per image
datasety degrade -i ./images -o ./degraded --type random --num-variants 3 --intensity-range 0.3-0.8
```

[Full documentation →](https://kontextox.github.io/datasety/commands/degrade)

---

### `character` — Character Dataset Generation

Generate character datasets using LLM-generated prompts + text-to-image (FLUX.2-klein local or cloud API).

<!-- screenshot: character -->

```bash
datasety character --output ./dataset --llm-ollama llama3.2 --num-images 20
```

<details>
<summary>Options</summary>

| Option                    | Description                                        | Default                                 |
| ------------------------- | -------------------------------------------------- | --------------------------------------- |
| `--reference`, `-r`       | Reference face image(s) (optional, prompt context) |                                         |
| `--output`, `-o`          | Output directory                                   | required                                |
| `--num-images`, `-n`      | Number of images to generate                       | `10`                                    |
| `--model`                 | Model for generation (local HF or API model ID)    | `black-forest-labs/FLUX.2-klein-4b-fp8` |
| `--gguf`                  | GGUF path/URL for quantized loading                |                                         |
| `--image-api`             | Use OpenAI-compatible API for image generation     | off                                     |
| `--character-description` | Text description of the character                  |                                         |
| `--style`                 | Style guidance (e.g., `photorealistic`)            |                                         |
| `--prompts-only`          | Only generate prompts, skip images                 | off                                     |
| `--prompts-file`          | Load prompts from file instead of LLM              |                                         |
| `--llm-api`               | Use OpenAI-compatible API for prompts              |                                         |
| `--llm-ollama MODEL`      | Use local Ollama server for prompts                |                                         |
| `--llm-gguf PATH`         | Use local GGUF model for prompts                   |                                         |
| `--llm-model REPO`        | Use HuggingFace model for prompts                  |                                         |
| `--device`                | `auto`, `cpu`, `cuda`, `mps`                       | `auto`                                  |
| `--steps`                 | Inference steps                                    | `4`                                     |
| `--cfg-scale`             | Guidance scale                                     | `4.0`                                   |
| `--seed`                  | Random seed                                        |                                         |
| `--height`                | Output image height                                | `1024`                                  |
| `--width`                 | Output image width                                 | `1024`                                  |
| `--output-format`         | `png`, `jpg`, `webp`                               | `png`                                   |
| `--batch-size`            | Flush GPU memory every N images                    | `0` (off)                               |
| `--dry-run`               | Preview prompts without generating images          | off                                     |

</details>

```bash
# Generate with local pipeline + Ollama prompts
datasety character -o ./dataset --llm-ollama llama3.2 --num-images 20

# Cloud API for images (no GPU needed)
OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
  datasety character -o ./dataset --prompts-file prompts.txt \
  --image-api --model black-forest-labs/flux.2-klein-4b

# Preview prompts only
datasety character -o ./dataset --llm-api --prompts-only
```

[Full documentation →](https://kontextox.github.io/datasety/commands/character)

---

### `sweep` — Parameter Grid Search

Generate workflow YAML files with parameter grid combinations for synthetic editing. Computes the Cartesian product of sweep parameters.

<!-- screenshot: sweep -->

```bash
datasety sweep -i ./images -o ./sweep_output -p "add a winter hat" --steps 4,8,16 --cfg-scale 1.0,2.5,5.0
```

<details>
<summary>Options</summary>

| Option             | Description                              | Default      |
| ------------------ | ---------------------------------------- | ------------ |
| `--input`, `-i`    | Input images directory                   | required     |
| `--output`, `-o`   | Base output directory                    | required     |
| `--prompt`, `-p`   | Edit prompt                              | required     |
| `--steps`          | Comma-separated step values to sweep     |              |
| `--cfg-scale`      | Comma-separated CFG values to sweep      |              |
| `--true-cfg-scale` | Comma-separated true CFG values to sweep |              |
| `--strength`       | Comma-separated strength values to sweep |              |
| `--lora`           | Comma-separated LoRA specs to sweep      |              |
| `--model`          | Comma-separated model names to sweep     |              |
| `--seed`           | Random seed (passed through)             |              |
| `--output-file`    | Output YAML path                         | `sweep.yaml` |
| `--run`            | Generate and immediately execute         | off          |

</details>

```bash
# Generate YAML, inspect, then run
datasety sweep -i ./images -o ./sweep -p "add sunglasses" --steps 4,8,16 --cfg-scale 1.0,2.5
datasety workflow -f sweep.yaml

# Generate and run immediately
datasety sweep -i ./images -o ./sweep -p "add a hat" --steps 4,8 --cfg-scale 2.0,3.0 --run
```

[Full documentation →](https://kontextox.github.io/datasety/commands/sweep)

---

### `train` — LoRA Fine-Tuning

Train a LoRA adapter for image generation models from a local dataset of image + caption pairs. Supports **FLUX.2-klein** (flow-matching) and **SDXL** (DDPM).

> **Use the base (undistilled) model for training.** Distilled models (`FLUX.2-klein-4B`) are for inference only. Use `FLUX.2-klein-base-4B` for LoRA training.

<!-- screenshot: train -->

```bash
datasety train --input ./dataset --output lora.safetensors --steps 500
```

<details>
<summary>Options</summary>

| Option               | Description                                  | Default                                  |
| -------------------- | -------------------------------------------- | ---------------------------------------- |
| `--input`, `-i`      | Dataset directory (images + `.txt` captions) | required                                 |
| `--output`, `-o`     | Output LoRA `.safetensors` path              | `lora.safetensors`                       |
| `--model`, `-m`      | HuggingFace base model repo ID               | `black-forest-labs/FLUX.2-klein-base-4B` |
| `--family`           | Model family: `flux`, `sdxl`                 | auto-detected                            |
| `--steps`            | Number of training steps                     | `100`                                    |
| `--lr`               | Learning rate                                | `1e-4`                                   |
| `--lora-rank`        | LoRA rank (4–64)                             | `16`                                     |
| `--lora-alpha`       | LoRA alpha                                   | `16.0`                                   |
| `--lora-dropout`     | LoRA dropout rate                            | `0.0`                                    |
| `--image-size`       | Training resolution (square crop)            | `512`                                    |
| `--device`           | `auto`, `cpu`, `cuda`, `mps`                 | `auto`                                   |
| `--seed`             | Random seed                                  | `42`                                     |
| `--save-every`       | Save checkpoint every N steps                | end only                                 |
| `--resume`           | Resume from a LoRA checkpoint (.safetensors) |                                          |
| `--validation-split` | Fraction of dataset for validation (0.0-0.5) |                                          |

</details>

```bash
# Prepare dataset
datasety resize -i ./raw -o ./dataset -r 512x512
datasety caption -i ./dataset -o ./dataset --trigger-word "ohwx person,"

# Train FLUX.2-klein LoRA (~8 GB VRAM)
datasety train \
    --input ./dataset \
    --output lora/flux_lora.safetensors \
    --model black-forest-labs/FLUX.2-klein-base-4B \
    --steps 500 --lr 1e-4 --lora-rank 16

# Use the trained LoRA
datasety synthetic --input-image photo.jpg --output-image out.png \
    --prompt "ohwx person wearing sunglasses" \
    --lora lora/flux_lora.safetensors:0.8

# SDXL LoRA
datasety train \
    --input ./dataset \
    --output sdxl_lora.safetensors \
    --model stabilityai/stable-diffusion-xl-base-1.0 \
    --family sdxl --steps 500 --image-size 1024
```

[Full documentation →](https://kontextox.github.io/datasety/commands/train)

---

### `workflow` — Multi-Step Pipelines

Run multi-step datasety pipelines from YAML or JSON files with dry-run validation.

<!-- screenshot: workflow -->

```bash
datasety workflow --file datasety.yaml --dry-run
```

<details>
<summary>Options</summary>

| Option         | Description                      | Default     |
| -------------- | -------------------------------- | ----------- |
| `--file`, `-f` | Path to workflow file            | auto-detect |
| `--dry-run`    | Validate steps without executing | off         |

</details>

Create `datasety.yaml`:

```yaml
steps:
  - command: resize
    args:
      input: ./raw
      output: ./resized
      resolution: 768x1024
  - command: caption
    args:
      input: ./resized
      output: ./resized
      llm-api: true
      model: gpt-5-nano
```

```bash
# Validate first, then execute
datasety workflow --dry-run
datasety workflow
```

[Full documentation →](https://kontextox.github.io/datasety/commands/workflow)

---

## License

MIT
