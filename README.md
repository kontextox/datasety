# datasety

CLI tool for dataset preparation: image resizing and captioning with Florence-2.

## Installation

```bash
pip install datasety
```

For captioning support (requires PyTorch and Transformers):

```bash
pip install datasety[caption]
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
4. Crops from the specified area to exact dimensions
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

## Common Workflows

### Prepare a LoRA Training Dataset

```bash
# 1. Resize images to 1024x1024
datasety resize -i ./raw -o ./dataset -r 1024x1024 --crop-position center

# 2. Generate captions with trigger word
datasety caption -i ./dataset -o ./dataset --trigger-word "[trigger]" --device cuda
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
- PyTorch + Transformers (for caption, install with `pip install datasety[caption]`)

## License

MIT
