# inspect

Show dataset statistics, caption coverage, and detect duplicates.

## Usage

```bash
datasety inspect --input ./dataset
```

## Options

| Option              | Description                                        | Default    |
| ------------------- | -------------------------------------------------- | ---------- |
| `--input`, `-i`     | Input directory                                    | (required) |
| `--duplicates`      | Detect duplicate/near-duplicate images             | `false`    |
| `--json`            | Export report as JSON to this path                  |            |
| `--csv`             | Export per-image data as CSV to this path           |            |
| `--recursive`, `-R` | Search input directory recursively                 | `false`    |

## Report Contents

- **Image count** and error count
- **Resolution**: min, max, average, unique sizes distribution
- **Orientation**: landscape, portrait, square breakdown
- **Formats**: file extension distribution
- **File sizes**: total and average
- **Caption coverage**: how many images have `.txt` files, average caption length, empty captions
- **Duplicates** (with `--duplicates`): groups of perceptually similar images using average hash with hamming distance threshold

## Examples

```bash
# Basic dataset report
datasety inspect -i ./dataset

# With duplicate detection
datasety inspect -i ./dataset --duplicates

# Export report to JSON
datasety inspect -i ./dataset --json report.json

# Export per-image data to CSV
datasety inspect -i ./dataset --csv images.csv

# Recursive scan with full export
datasety inspect -i ./dataset -R --duplicates --json report.json --csv images.csv
```

## JSON Output Format

```json
{
  "path": "./dataset",
  "total_images": 150,
  "errors": 0,
  "resolution": {
    "min": "512x512",
    "max": "2048x2048",
    "average": "1024x1024"
  },
  "formats": { "jpg": 120, "png": 30 },
  "orientation": { "landscape": 45, "square": 80, "portrait": 25 },
  "total_size_mb": 245.3,
  "captions_found": 140,
  "captions_missing": 10,
  "avg_caption_length": 85
}
```

## CSV Output Format

| Column       | Description                    |
| ------------ | ------------------------------ |
| `file`       | Image file path                |
| `width`      | Image width in pixels          |
| `height`     | Image height in pixels         |
| `format`     | File extension                 |
| `size_kb`    | File size in KB                |
| `has_caption`| Whether a .txt file exists     |
