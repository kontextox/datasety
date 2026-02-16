# resize

Resize and crop images to a target resolution.

## Usage

```bash
datasety resize --input ./images --output ./resized --resolution 768x1024
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--input`, `-i` | Input directory | (required*) |
| `--output`, `-o` | Output directory | (required*) |
| `--input-image` | Single input image | |
| `--output-image` | Single output image | |
| `--resolution`, `-r` | Target resolution (WIDTHxHEIGHT) | (required) |
| `--crop-position` | `top`, `center`, `bottom`, `left`, `right` | `center` |
| `--input-format` | Comma-separated formats | `jpg,jpeg,png,webp` |
| `--output-format` | `jpg`, `png`, `webp` | `jpg` |
| `--output-name-numbers` | Rename to 1.jpg, 2.jpg, ... | `false` |

## Examples

```bash
# Batch resize with top crop
datasety resize -i ./raw -o ./dataset -r 1024x1024 --crop-position top

# Single image
datasety resize --input-image photo.jpg --output-image resized.jpg -r 512x512

# Sequential numbering
datasety resize -i ./photos -o ./processed -r 768x1024 --output-name-numbers
```

## How It Works

1. Finds all images matching input formats
2. Skips images smaller than target resolution
3. Resizes proportionally so the smaller side matches target
4. Crops from the specified position to exact dimensions
5. Saves with high quality (95% for jpg/webp)
