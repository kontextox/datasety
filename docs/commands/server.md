# server

Start the universal dataset management web dashboard. Replaces the former `--server` flags on `inspect` and `align`.

## Usage

```bash
datasety server --input ./dataset
datasety server --input ./target --control ./control
```

## Options

| Option              | Description                                           | Default    |
| ------------------- | ----------------------------------------------------- | ---------- |
| `--input`, `-i`     | Dataset directory to manage                           | (required) |
| `--control`, `-c`   | Control images directory (enables Pairs mode)         |            |
| `--port`            | Port for the web server                               | `8080`     |
| `--recursive`, `-R` | Search directories recursively for images             | `false`    |
| `--duplicates`      | Pre-compute perceptual hashes for duplicate detection | `false`    |

## Dashboard Tabs

### Gallery

Browse all images in a scrollable grid with thumbnails.

- **Sort** by filename, size, or resolution
- **Filter** by format, orientation, or caption status
- **Click** any image to open a full detail view (file info, caption editor, delete)
- **Upload** new images directly from the browser

### Compare

Side-by-side image comparison with a drag slider.

- Drag the divider to reveal left/right images
- Select any two images from the gallery to compare

### Pairs _(visible when `--control` is set)_

Visual comparison of control/target image pairs — the equivalent of the former `datasety align --server`.

- Navigate pairs with **Prev / Next** or arrow keys
- **Compare slider** to drag between control and target
- **Caption editors** for both control and target sides
- **Save captions** with `Ctrl+S`
- **Delete pair** removes both images and their caption files

### Stats

Live dataset statistics updated after every mutation:

- Image count, total size
- Resolution distribution
- Format breakdown
- Orientation breakdown (landscape / square / portrait)
- Caption coverage

## Examples

```bash
# Gallery + stats for a dataset
datasety server -i ./dataset

# With duplicate detection pre-computed
datasety server -i ./dataset --duplicates

# Pairs comparison (align workflow)
datasety server -i ./target --control ./control

# Custom port, recursive scan
datasety server -i ./dataset --port 9000 -R
```

## Keyboard Shortcuts

| Key       | Action                        |
| --------- | ----------------------------- |
| `←` / `→` | Previous / next image or pair |
| `Escape`  | Close modal / panel           |
| `Ctrl+S`  | Save caption                  |
| `T`       | Toggle dark / light theme     |
| `?`       | Show keyboard help            |
