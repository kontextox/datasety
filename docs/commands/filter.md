# filter

Filter, curate, or clean datasets based on image content using CLIP (arbitrary text queries) or NudeNet (NSFW label detection).

## Usage

```bash
datasety filter --input ./dataset --output ./rejected --query "leg,male face" --action move
```

## Supported Models

| Model              | Description                                        | Speed     |
| ------------------ | -------------------------------------------------- | --------- |
| `clip` (default)   | CLIP ViT-B/32 — zero-shot classification, any text | ~50ms/img |
| `nudenet`          | NudeNet ONNX — NSFW detection, 18 fixed labels     | ~10ms/img |

### NudeNet Labels

```
FEMALE_GENITALIA_COVERED, FACE_FEMALE, BUTTOCKS_EXPOSED,
FEMALE_BREAST_EXPOSED, FEMALE_GENITALIA_EXPOSED, MALE_BREAST_EXPOSED,
ANUS_EXPOSED, FEET_EXPOSED, BELLY_COVERED, FEET_COVERED,
ARMPITS_COVERED, ARMPITS_EXPOSED, FACE_MALE, BELLY_EXPOSED,
MALE_GENITALIA_EXPOSED, ANUS_COVERED, FEMALE_BREAST_COVERED,
BUTTOCKS_COVERED
```

## Actions

| Action   | Behavior                                                       |
| -------- | -------------------------------------------------------------- |
| `move`   | Move matching images to `--output` (default)                   |
| `copy`   | Copy matching images to `--output`                             |
| `delete` | Permanently delete matching images (requires `--confirm`)      |
| `keep`   | Inverse — remove/move everything that does NOT match           |

With `--action keep`:
- If `--output` is set, non-matching images are moved there (safe).
- Without `--output`, non-matching images are deleted (requires `--confirm`).

Companion files (`.txt`, `.caption`, `.json`) are always handled alongside their images.

## Options

| Option                 | Description                                           | Default    |
| ---------------------- | ----------------------------------------------------- | ---------- |
| `--input`, `-i`        | Input directory                                       | (required) |
| `--output`, `-o`       | Output directory for matched/rejected images          |            |
| `--query`, `-q`        | Comma-separated text queries (CLIP)                   |            |
| `--labels`, `-l`       | Comma-separated NudeNet labels                        |            |
| `--model`              | `clip` or `nudenet`                                   | `clip`     |
| `--action`             | `move`, `copy`, `delete`, or `keep`                   | `move`     |
| `--threshold`          | Confidence threshold (0.0-1.0)                        | `0.5`      |
| `--device`             | `auto`, `cpu`, `cuda`, or `mps`                       | `auto`     |
| `--confirm`            | Required for destructive actions                      | `false`    |
| `--preserve-structure` | Keep subfolder hierarchy in output (with `--recursive`) | `false`    |
| `--log`                | Write CSV log of all decisions to this path           |            |
| `--dry-run`            | Preview detections without modifying files            | `false`    |
| `--recursive`, `-R`    | Search input directory recursively                    | `false`    |

## Examples

```bash
# Move images with legs or male faces to a reject folder
datasety filter -i ./dataset -o ./rejected --query "leg,male face" --action move

# Delete NSFW images using NudeNet
datasety filter -i ./dataset --labels "FEMALE_BREAST_EXPOSED,MALE_GENITALIA_EXPOSED" \
    --action delete --model nudenet --threshold 0.6 --confirm

# Keep only images matching "hat and socks", move the rest
datasety filter -i ./dataset -o ./rejected --query "hat and socks" --action keep

# Preview without changes
datasety filter -i ./dataset --query "blurry,low quality" --action delete --dry-run -R

# Scan nested folders, preserve structure, write log
datasety filter -i ./dataset -o ./filtered --query "outdoor scene" \
    --action copy -R --preserve-structure --log filter_log.csv
```
