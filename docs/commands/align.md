# align

Align control/target image pairs for training compatibility. Ensures matching dimensions, multiples of 32, and consistent formats.

## Usage

```bash
datasety align --target ./target --control ./control --dry-run
```

## Options

| Option              | Description                                    | Default         |
| ------------------- | ---------------------------------------------- | --------------- |
| `--target`, `-t`    | Target images directory                        | (required)      |
| `--control`, `-c`   | Control images directory                       | (required)      |
| `--multiple-of`     | Align dimensions to this multiple              | `32`            |
| `--output-format`   | Convert images to format: `jpg`, `png`, `webp` | (keep original) |
| `--recursive`, `-R` | Search input directories recursively           | `false`         |
| `--dry-run`         | Preview changes without modifying files        | `false`         |

## Examples

```bash
# Preview fixes
datasety align -t ./target -c ./control --dry-run

# Apply fixes
datasety align -t ./target -c ./control

# Fix and convert to jpg
datasety align -t ./target -c ./control --output-format jpg
```

> **Visual comparison:** use [`datasety server --control`](/commands/server) to browse and compare aligned pairs in the browser.

## How It Works

1. Matches pairs by filename stem (e.g., `001.jpg` ↔ `001.png`)
2. Crops target dimensions to nearest multiple of 32 (center crop)
3. Resizes control images to match target dimensions (LANCZOS)
4. Optionally converts all images to a single format
5. Reports missing pairs, orphan controls, and dimension issues
