# align

Align control/target image pairs for training compatibility. Ensures matching dimensions, multiples of 32, and consistent formats. Includes a built-in web server for visual comparison.

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
| `--server`          | Start web server for visual comparison         | `false`         |
| `--port`            | Port for the comparison web server             | `8787`          |

## Examples

```bash
# Preview fixes
datasety align -t ./target -c ./control --dry-run

# Apply fixes
datasety align -t ./target -c ./control

# Fix and convert to jpg
datasety align -t ./target -c ./control --output-format jpg

# Visual comparison web UI
datasety align -t ./target -c ./control --server
datasety align -t ./target -c ./control --server --port 9000
```

## How It Works

1. Matches pairs by filename stem (e.g., `001.jpg` ↔ `001.png`)
2. Crops target dimensions to nearest multiple of 32 (center crop)
3. Resizes control images to match target dimensions (LANCZOS)
4. Optionally converts all images to a single format
5. Reports missing pairs, orphan controls, and dimension issues

## Web Server

Use `--server` to start a local web UI for visually comparing aligned pairs.

- **Compare slider** — drag to reveal control vs target side by side
- **Caption editing** — view and edit `.txt` caption files for both sides; saving an empty caption deletes the file
- **Delete pairs** — remove image pairs and associated caption files
- **Keyboard shortcuts** — arrow keys to navigate, `[`/`]` to move the slider, `Ctrl+S` to save, `?` for help
- **Responsive** — on wide screens, captions appear beside the image; on mobile, they stack below
- **Light/dark theme** — toggle via the header button
