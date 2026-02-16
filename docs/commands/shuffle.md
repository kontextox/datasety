# shuffle

Generate random captions by picking one variant from each text group.

## Usage

```bash
datasety shuffle -i ./images -o ./captions \
    --group "Hello.|Hey!|Bonjour." \
    --group "World.|Earth!"
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--input`, `-i` | Input directory containing images | (required) |
| `--output`, `-o` | Output directory for .txt files | (required) |
| `--group`, `-g` | Inline `\|`-separated, .txt file, or URL | (required) |
| `--separator` | Separator between groups | `" "` |
| `--seed` | Random seed for reproducibility | (random) |
| `--dry-run` | Preview captions without writing | `false` |
| `--show-distribution` | Show caption distribution | `false` |

## Group Sources

- **Inline**: `"Hello.|Hey!|Bonjour."` (pipe-separated)
- **File**: `phrases.txt` (one variant per line)
- **URL**: `https://example.com/phrases.txt` (fetched, one per line)

## Examples

```bash
# Inline groups
datasety shuffle -i ./images -o ./captions \
    --group "A photo of a person.|Portrait of someone." \
    --group "Remove the hat.|Take off the hat."

# Mix file and inline
datasety shuffle -i ./images -o ./captions \
    --group subjects.txt \
    --group "ending A|ending B" \
    --seed 42 --show-distribution
```
