# datasety

CLI tool for dataset preparation: resize, align, caption, shuffle, synthetic, and mask generation.

## Installation

```bash
pip install datasety
```

Install with specific features:

```bash
pip install datasety[caption]     # Florence-2 captioning
pip install datasety[synthetic]   # Image editing (Qwen, FLUX, SDXL, Hunyuan)
pip install datasety[mask]        # Mask generation (SAM 3, Grounded SAM 2, CLIPSeg)
pip install datasety[all]         # All features
```

## Usage

### Resize Images

Resize and crop images to a target resolution:

```bash
datasety resize --input ./images --output ./resized --resolution 768x1024
```

**Options:**

| Option                  | Description                                               | Default             |
| ----------------------- | --------------------------------------------------------- | ------------------- |
| `--input`, `-i`         | Input directory                                           | (required\*)        |
| `--output`, `-o`        | Output directory                                          | (required\*)        |
| `--input-image`         | Single input image (alternative to `--input`)             |                     |
| `--output-image`        | Single output image (use with `--input-image`)            |                     |
| `--resolution`, `-r`    | Target resolution (WIDTHxHEIGHT)                          | (required)          |
| `--crop-position`       | Crop position: `top`, `center`, `bottom`, `left`, `right` | `center`            |
| `--input-format`        | Comma-separated formats                                   | `jpg,jpeg,png,webp` |
| `--output-format`       | Output format: `jpg`, `png`, `webp`                       | `jpg`               |
| `--output-name-numbers` | Rename files to 1.jpg, 2.jpg, ...                         | `false`             |

**Example:**

```bash
datasety resize \
    --input ./raw_photos \
    --output ./dataset \
    --resolution 1024x1024 \
    --crop-position top \
    --output-format jpg \
    --output-name-numbers

# Single-image mode
datasety resize \
    --input-image photo.jpg \
    --output-image resized.jpg \
    --resolution 512x512
```

**How it works:**

1. Finds all images matching input formats
2. Skips images where either dimension is smaller than target
3. Resizes proportionally so the smaller side matches target
4. Crops from the specified position to exact dimensions
5. Saves with high quality (95% for jpg/webp)

### Generate Captions

Generate captions for images using Microsoft's Florence-2 model:

```bash
datasety caption --input ./images --output ./captions --florence-2-large
```

**Options:**

| Option               | Description                                        | Default                   |
| -------------------- | -------------------------------------------------- | ------------------------- |
| `--input`, `-i`      | Input directory                                    | (required\*)              |
| `--output`, `-o`     | Output directory for .txt files                    | (required\*)              |
| `--input-image`      | Single input image (alternative to `--input`)      |                           |
| `--output-caption`   | Single output .txt path (use with `--input-image`) |                           |
| `--device`           | `auto`, `cpu`, or `cuda`                           | `auto`                    |
| `--trigger-word`     | Text to prepend to captions                        | (none)                    |
| `--prompt`           | Florence-2 task prompt                             | `<MORE_DETAILED_CAPTION>` |
| `--model`            | Any HuggingFace model (overrides base/large flags) | (none)                    |
| `--num-beams`        | Beam search width (1 = greedy)                     | `3`                       |
| `--florence-2-base`  | Use base model (0.23B, faster)                     |                           |
| `--florence-2-large` | Use large model (0.77B, better)                    | (default)                 |

**Available prompts:**

- `<CAPTION>` - Brief caption
- `<DETAILED_CAPTION>` - Detailed caption
- `<MORE_DETAILED_CAPTION>` - Most detailed caption (default)

**Examples:**

```bash
datasety caption \
    --input ./dataset \
    --output ./dataset \
    --device cuda \
    --trigger-word "photo of sks person," \
    --florence-2-large

# Use a custom model
datasety caption \
    --input ./dataset \
    --output ./dataset \
    --device cuda \
    --model "microsoft/Florence-2-large"

# Single-image mode
datasety caption \
    --input-image photo.jpg \
    --output-caption photo.txt \
    --device cuda
```

This creates a `.txt` file for each image with the generated caption.

### Align Control/Target Pairs

Align control and target image pairs for training (e.g., ai-toolkit LoRA with control images). Ensures matching dimensions, multiples of 32, and consistent formats:

```bash
datasety align --target ./target --control ./control --dry-run
```

**Options:**

| Option            | Description                                        | Default         |
| ----------------- | -------------------------------------------------- | --------------- |
| `--target`, `-t`  | Target images directory                            | (required)      |
| `--control`, `-c` | Control images directory                           | (required)      |
| `--multiple-of`   | Align dimensions to this multiple                  | `32`            |
| `--output-format` | Convert all images to format: `jpg`, `png`, `webp` | (keep original) |
| `--dry-run`       | Preview changes without modifying files            | `false`         |

**Examples:**

```bash
# Preview what needs fixing
datasety align -t ./target -c ./control --dry-run

# Fix all pairs in place
datasety align -t ./target -c ./control

# Fix and convert everything to jpg
datasety align -t ./target -c ./control --output-format jpg
```

**How it works:**

1. Matches pairs by filename stem (target `001.jpg` ↔ control `001.png`)
2. Crops target dimensions to the nearest multiple of 32 (center crop)
3. Resizes control images to match target dimensions (LANCZOS)
4. Optionally converts all images to a single format
5. Reports missing pairs, orphan controls, and dimension issues

### Shuffle Captions

Generate random captions by picking one variant from each text group:

```bash
datasety shuffle -i ./images -o ./captions \
    --group "Hello.|Hey!|Bonjour." \
    --group "How to..|Wow.." \
    --group "Foo Bar!"
```

Each `--group` picks one random variant per image. Groups support three sources:

- **Inline**: `"Hello.|Hey!|Bonjour."` (pipe-separated)
- **File**: `phrases.txt` (one variant per line)
- **URL**: `https://example.com/phrases.txt` (fetched, one variant per line)

**Options:**

| Option                | Description                                     | Default    |
| --------------------- | ----------------------------------------------- | ---------- |
| `--input`, `-i`       | Input directory containing images               | (required) |
| `--output`, `-o`      | Output directory for .txt files                 | (required) |
| `--group`, `-g`       | Inline `\|`-separated, `.txt` file path, or URL | (required) |
| `--separator`         | Separator between groups                        | `" "`      |
| `--seed`              | Random seed for reproducibility                 | (random)   |
| `--dry-run`           | Preview captions without writing files          | `false`    |
| `--show-distribution` | Show caption distribution after generation      | `false`    |

**Examples:**

```bash
# Inline groups
datasety shuffle -i ./images -o ./captions \
    --group "Hello.|Hey!|Bonjour." \
    --group "How to..|Wow.." \
    --group "Foo Bar!"

# Mix file, URL, and inline
datasety shuffle -i ./images -o ./captions \
    --group starts.txt \
    --group https://example.com/middles.txt \
    --group "ending A|ending B"
```

**Example output:**

- `Hello. Wow.. Foo Bar!`
- `Bonjour. How to.. Foo Bar!`
- `Hey! Wow.. Foo Bar!`

### Generate Synthetic Images

Generate synthetic variations of images using image editing models:

```bash
datasety synthetic --input ./images --output ./synthetic --prompt "add a winter hat" --steps 4
```

The default model is **FLUX.2-klein** (fast, 8 GB VRAM). The model family is auto-detected from the `--model` name:

| Model Family                 | Example Model IDs                          | Key Params                   |
| ---------------------------- | ------------------------------------------ | ---------------------------- |
| **FLUX.2 klein** (default)   | `black-forest-labs/FLUX.2-klein-4B`        | `--strength`, steps=4        |
| **FLUX.2 dev**               | `black-forest-labs/FLUX.2-dev`             | `--strength`, steps=28       |
| **Qwen**                     | `Qwen/Qwen-Image-Edit-2511`                | `--true-cfg-scale`, steps=40 |
| **FireRed** (uses Qwen pipe) | `FireRedTeam/FireRed-Image-Edit-1.0`       | `--true-cfg-scale`, steps=40 |
| **FLUX Kontext**             | `black-forest-labs/FLUX.1-Kontext-dev`     | steps=28                     |
| **LongCat**                  | `meituan-longcat/LongCat-Image-Edit-Turbo` | steps=50                     |
| **SDXL**                     | `stabilityai/stable-diffusion-xl-*`        | `--strength`, steps=30       |
| **HunyuanImage**             | `tencent/HunyuanImage-3.0`                 | steps=50                     |

**Options:**

| Option              | Description                                    | Default                             |
| ------------------- | ---------------------------------------------- | ----------------------------------- |
| `--input`, `-i`     | Input directory                                | (required\*)                        |
| `--output`, `-o`    | Output directory                               | (required\*)                        |
| `--input-image`     | Single input image (alternative to `--input`)  |                                     |
| `--output-image`    | Single output image (use with `--input-image`) |                                     |
| `--prompt`, `-p`    | Edit prompt                                    | (required)                          |
| `--model`           | Model to use (auto-detects family)             | `black-forest-labs/FLUX.2-klein-4B` |
| `--weights`         | Fine-tuned weights as `repo_id:filename`       | (none)                              |
| `--device`          | `auto`, `cpu`, or `cuda`                       | `auto`                              |
| `--cpu-offload`     | Offload model to CPU when not in use           | `false`                             |
| `--steps`           | Number of inference steps                      | `40`                                |
| `--cfg-scale`       | Guidance scale                                 | `1.0`                               |
| `--true-cfg-scale`  | True CFG scale (Qwen only)                     | `4.0`                               |
| `--negative-prompt` | Negative prompt                                | `" "`                               |
| `--num-images`      | Images to generate per input                   | `1`                                 |
| `--seed`            | Random seed for reproducibility                | (random)                            |
| `--gguf`            | Path/URL to GGUF file for quantized loading    | (none)                              |
| `--strength`        | Img2img strength for SDXL/FLUX.2 (0.0-1.0)     | `0.7`                               |
| `--output-format`   | Output format: `png`, `jpg`, `webp`            | `png`                               |

**Examples:**

```bash
# FLUX.2 klein (default — fast, 8 GB VRAM)
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --prompt "add a winter hat" \
    --steps 4 \
    --cfg-scale 2.5

# Single-image mode
datasety synthetic \
    --input-image photo.jpg \
    --output-image edited.png \
    --prompt "add sunglasses" \
    --steps 4

# FLUX.2 dev (higher quality, 24 GB VRAM)
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --model "black-forest-labs/FLUX.2-dev" \
    --prompt "add a winter hat" \
    --strength 0.7 \
    --steps 28

# Qwen
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --model "Qwen/Qwen-Image-Edit-2511" \
    --prompt "add sunglasses to the person" \
    --device cuda \
    --steps 40 \
    --true-cfg-scale 4.0

# FLUX Kontext with GGUF quantization (auto-downloads from HF URL)
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --model "black-forest-labs/FLUX.1-Kontext-dev" \
    --gguf "https://huggingface.co/unsloth/FLUX.1-Kontext-dev-GGUF/resolve/main/flux1-kontext-dev-Q4_K_S.gguf" \
    --prompt "add a winter hat" \
    --cpu-offload

# LongCat
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --model "meituan-longcat/LongCat-Image-Edit-Turbo" \
    --prompt "add a winter hat" \
    --steps 50

# SDXL img2img
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --model "stabilityai/stable-diffusion-xl-base-1.0" \
    --prompt "a photo with a red scarf, high quality" \
    --strength 0.7 \
    --cfg-scale 7.5 \
    --steps 30

# Qwen with fine-tuned weights (fewer steps, less VRAM)
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --model "Qwen/Qwen-Image-Edit-2511" \
    --weights "Phr00t/Qwen-Image-Edit-Rapid-AIO:v23/Qwen-Rapid-AIO-NSFW-v23.safetensors" \
    --prompt "add a winter hat" \
    --steps 20 \
    --output-format jpg
```

### Generate Masks

Generate binary masks from images using text keywords for LoRA training:

```bash
datasety mask --input ./dataset --output ./masks --keywords "face,hair"
```

Masks are white (255) for the region of interest and black (0) for ignore. Multiple keywords are combined (union) into a single mask.

**Supported models:**

| Model            | Description                                              |
| ---------------- | -------------------------------------------------------- |
| `sam3` (default) | SAM 3 — native text prompting, best quality              |
| `grounded-sam2`  | Grounding DINO + SAM 2 — most battle-tested              |
| `clipseg`        | CLIPSeg — lightweight, no extra deps beyond transformers |

**Options:**

| Option             | Description                                              | Default      |
| ------------------ | -------------------------------------------------------- | ------------ |
| `--input`, `-i`    | Input directory containing images                        | (required\*) |
| `--output`, `-o`   | Output directory for mask images                         | (required\*) |
| `--input-image`    | Single input image (alternative to `--input`)            |              |
| `--output-image`   | Single output mask path (use with `--input-image`)       |              |
| `--keywords`, `-k` | Comma-separated keywords (e.g., `"face,hair,hat"`)       | (required)   |
| `--model`          | Segmentation model: `sam3`, `grounded-sam2`, `clipseg`   | `sam3`       |
| `--device`         | `auto`, `cpu`, or `cuda`                                 | `auto`       |
| `--threshold`      | Confidence threshold (0.0-1.0)                           | `0.3`        |
| `--padding`        | Pixels to expand mask by (dilation)                      | `0`          |
| `--blur`           | Gaussian blur radius for mask edges (0=sharp)            | `0`          |
| `--invert`         | Invert mask (black=ROI, white=ignore)                    | `false`      |
| `--naming`         | `folder` (same name in output dir) or `suffix` (`_mask`) | `folder`     |
| `--output-format`  | `png`, `jpg`, `webp`                                     | `png`        |
| `--dry-run`        | Preview detections without saving masks                  | `false`      |

**Examples:**

```bash
# SAM 3 (default, best quality)
datasety mask \
    --input ./dataset \
    --output ./masks \
    --keywords "face,hair" \
    --device cuda

# CLIPSeg (lightweight, CPU-friendly)
datasety mask \
    --input ./dataset \
    --output ./masks \
    --keywords "face" \
    --model clipseg \
    --threshold 0.5 \
    --device cpu

# Grounded SAM 2 with mask refinement
datasety mask \
    --input ./dataset \
    --output ./masks \
    --keywords "hat,glasses" \
    --model grounded-sam2 \
    --padding 5 \
    --blur 3

# Suffix mode (saves masks alongside source images)
datasety mask \
    --input ./dataset \
    --output ./dataset \
    --keywords "face" \
    --naming suffix

# Preview without saving
datasety mask \
    --input ./dataset \
    --output ./masks \
    --keywords "person" \
    --dry-run

# Single-image mode
datasety mask \
    --input-image photo.jpg \
    --output-image photo_mask.png \
    --keywords "face,hair"
```

**Output formats:**

- **Folder mode** (default): `./masks/photo.png` for `./dataset/photo.jpg` — use with ai-toolkit `mask_path` folder config
- **Suffix mode**: `./dataset/photo_mask.png` alongside source images

## Common Workflows

### Prepare a LoRA Training Dataset

```bash
# 1. Resize images to 1024x1024
datasety resize -i ./raw -o ./dataset -r 1024x1024 --crop-position center

# 2. Generate captions with trigger word
datasety caption -i ./dataset -o ./dataset --trigger-word "[trigger]" --device cuda
```

### Prepare Control/Target Pairs for LoRA Training

```bash
# 1. Align pairs (dimensions to multiple of 32, match sizes)
datasety align -t ./target -c ./control --dry-run
datasety align -t ./target -c ./control

# 2. Generate captions for target images
datasety caption -i ./target -o ./target --device cuda
```

### Generate Varied Captions for Training

```bash
# Using inline groups
datasety shuffle -i ./dataset -o ./dataset \
    --group "A photo of a person.|Portrait of someone.|Image of a figure." \
    --group "Remove the hat.|Take off the hat.|Strip the hat away." \
    --group "Show natural ears.|Reveal the ears.|Expose realistic ears." \
    --seed 42 --show-distribution

# Using text files (one variant per line)
datasety shuffle -i ./dataset -o ./dataset \
    --group subjects.txt \
    --group actions.txt \
    --group details.txt \
    --seed 42
```

### Augment Dataset with Synthetic Variations

```bash
# Generate variations with different accessories
datasety synthetic \
    -i ./dataset \
    -o ./synthetic \
    --prompt "add a red scarf" \
    --num-images 2 \
    --device cuda
```

### Generate Masks for Focused LoRA Training

```bash
# 1. Generate face masks for your dataset
datasety mask -i ./dataset -o ./masks -k "face" --device cuda

# 2. Use with ai-toolkit: set mask_path to ./masks in your config
```

### Batch Process with Numbered Files

```bash
datasety resize \
    -i ./photos \
    -o ./processed \
    -r 768x1024 \
    --output-name-numbers \
    --crop-position top
```

## Requirements

- Python 3.10+
- Pillow (for resize, align, shuffle)
- PyTorch + Transformers (for caption: `pip install datasety[caption]`)
- PyTorch + Diffusers (for synthetic: `pip install datasety[synthetic]`)
- PyTorch + Transformers + SAM 2 (for mask: `pip install datasety[mask]`)

## License

MIT
