# datasety

CLI tool for dataset preparation: resize, caption, and synthetic image generation.

## Installation

```bash
pip install datasety
```

Install with specific features:

```bash
pip install datasety[caption]     # Florence-2 captioning
pip install datasety[synthetic]   # Qwen image editing
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
| `--input`, `-i`         | Input directory                                           | (required)          |
| `--output`, `-o`        | Output directory                                          | (required)          |
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

| Option               | Description                     | Default                   |
| -------------------- | ------------------------------- | ------------------------- |
| `--input`, `-i`      | Input directory                 | (required)                |
| `--output`, `-o`     | Output directory for .txt files | (required)                |
| `--device`           | `cpu` or `cuda`                 | `cpu`                     |
| `--trigger-word`     | Text to prepend to captions     | (none)                    |
| `--prompt`           | Florence-2 task prompt          | `<MORE_DETAILED_CAPTION>` |
| `--florence-2-base`  | Use base model (0.23B, faster)  |                           |
| `--florence-2-large` | Use large model (0.77B, better) | (default)                 |

**Available prompts:**

- `<CAPTION>` - Brief caption
- `<DETAILED_CAPTION>` - Detailed caption
- `<MORE_DETAILED_CAPTION>` - Most detailed caption (default)

**Example:**

```bash
datasety caption \
    --input ./dataset \
    --output ./dataset \
    --device cuda \
    --trigger-word "photo of sks person," \
    --florence-2-large
```

This creates a `.txt` file for each image with the generated caption.

### Generate Synthetic Images

Generate synthetic variations of images using Qwen-Image-Edit:

```bash
datasety synthetic --input ./images --output ./synthetic --prompt "add a winter hat"
```

**Options:**

| Option              | Description                       | Default                    |
| ------------------- | --------------------------------- | -------------------------- |
| `--input`, `-i`     | Input directory                   | (required)                 |
| `--output`, `-o`    | Output directory                  | (required)                 |
| `--prompt`, `-p`    | Edit prompt                       | (required)                 |
| `--model`           | Model to use                      | `Qwen/Qwen-Image-Edit-2511`|
| `--device`          | `cpu` or `cuda`                   | `cuda`                     |
| `--steps`           | Number of inference steps         | `40`                       |
| `--cfg-scale`       | Guidance scale                    | `1.0`                      |
| `--true-cfg-scale`  | True CFG scale                    | `4.0`                      |
| `--negative-prompt` | Negative prompt                   | `" "`                      |
| `--num-images`      | Images to generate per input      | `1`                        |
| `--seed`            | Random seed for reproducibility   | (random)                   |

**Example:**

```bash
datasety synthetic \
    --input ./dataset \
    --output ./synthetic \
    --prompt "add sunglasses to the person, keep everything else the same" \
    --device cuda \
    --steps 40 \
    --true-cfg-scale 4.0 \
    --seed 42
```

## Common Workflows

### Prepare a LoRA Training Dataset

```bash
# 1. Resize images to 1024x1024
datasety resize -i ./raw -o ./dataset -r 1024x1024 --crop-position center

# 2. Generate captions with trigger word
datasety caption -i ./dataset -o ./dataset --trigger-word "[trigger]" --device cuda
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
- Pillow (for resize)
- PyTorch + Transformers (for caption: `pip install datasety[caption]`)
- PyTorch + Diffusers (for synthetic: `pip install datasety[synthetic]`)

## License

MIT
