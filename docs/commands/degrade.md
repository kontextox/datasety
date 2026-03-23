# degrade

Create degraded versions of images for upscale/enhance training. No extra dependencies — pure Pillow.

## Usage

```bash
datasety degrade --input ./originals --output ./dataset --type random --paired
```

## Degradation Types

| Type          | Effect at intensity=1.0                  |
| ------------- | ---------------------------------------- |
| `lowres`      | 16x bilinear downscale + nearest upscale |
| `oversharpen` | Aggressive unsharp mask (percent=1000)   |
| `noise`       | 50% Gaussian noise blend                 |
| `blur`        | Radius 3px Gaussian blur                 |
| `jpeg`        | Quality 5 JPEG compression               |
| `motion-blur` | Horizontal motion blur                   |
| `pixelate`    | 16x nearest-neighbor pixelation          |
| `color-bands` | 3-bit posterization                      |
| `upscale-sim` | 8x bilinear down + Lanczos up            |

## Options

| Option              | Description                       | Default      |
| ------------------- | --------------------------------- | ------------ |
| `--input`, `-i`     | Input directory                   | (required\*) |
| `--output`, `-o`    | Output directory                  | (required\*) |
| `--input-image`     | Single input image                |              |
| `--output-image`    | Single output image               |              |
| `--type`, `-t`      | Degradation type(s), repeatable   | `random`     |
| `--intensity`       | Global intensity 0.0-1.0          | `0.5`        |
| `--intensity-range` | Random range MIN-MAX              |              |
| `--chain`           | Apply multiple types sequentially | `false`      |
| `--num-variants`    | Variants per input image          | `1`          |
| `--paired`          | Create control/ + target/ subdirs | `false`      |
| `--seed`            | Random seed                       | (random)     |
| `--output-format`   | `png`, `jpg`, `webp`              | `png`        |
| `--skip-existing`   | Skip images with existing output  | `false`      |
| `--workers`         | Parallel workers for processing   | `1`          |
| `--progress`        | Show tqdm progress bar            | `false`      |
| `--dry-run`         | Preview without writing files     | `false`      |

## Examples

```bash
# Random degradation for upscale training
datasety degrade -i ./originals -o ./dataset --type random --intensity-range 0.2-0.8 --paired

# Chain specific degradations
datasety degrade -i ./images -o ./degraded --type jpeg --type noise --chain --paired

# Multiple variants per image
datasety degrade -i ./images -o ./degraded --type random --num-variants 3 --seed 42
```
