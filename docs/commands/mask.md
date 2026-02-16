# mask

Generate binary masks from images using text keywords.

## Usage

```bash
datasety mask --input ./dataset --output ./masks --keywords "face,hair"
```

## Supported Models

| Model | Description |
| --- | --- |
| `sam3` (default) | SAM 3 — native text prompting, best quality |
| `grounded-sam2` | Grounding DINO + SAM 2 — most battle-tested |
| `clipseg` | CLIPSeg — lightweight, no extra deps beyond transformers |

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--input`, `-i` | Input directory | (required*) |
| `--output`, `-o` | Output directory for masks | (required*) |
| `--input-image` | Single input image | |
| `--output-image` | Single output mask | |
| `--keywords`, `-k` | Comma-separated keywords | (required) |
| `--model` | `sam3`, `grounded-sam2`, `clipseg` | `sam3` |
| `--device` | `auto`, `cpu`, or `cuda` | `auto` |
| `--threshold` | Confidence threshold (0.0-1.0) | `0.3` |
| `--padding` | Pixels to expand mask (dilation) | `0` |
| `--blur` | Gaussian blur radius for edges | `0` |
| `--invert` | Invert mask colors | `false` |
| `--naming` | `folder` or `suffix` (_mask) | `folder` |
| `--output-format` | `png`, `jpg`, `webp` | `png` |
| `--dry-run` | Preview without saving | `false` |

## Examples

```bash
# SAM 3 (default)
datasety mask -i ./dataset -o ./masks -k "face,hair" --device cuda

# CLIPSeg (lightweight)
datasety mask -i ./dataset -o ./masks -k "face" --model clipseg --threshold 0.5

# Grounded SAM 2 with refinement
datasety mask -i ./dataset -o ./masks -k "hat,glasses" --model grounded-sam2 --padding 5 --blur 3

# Suffix mode (saves alongside source images)
datasety mask -i ./dataset -o ./dataset -k "face" --naming suffix
```
