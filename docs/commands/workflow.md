# workflow

Run multi-step datasety workflows from YAML or JSON files.

## Usage

```bash
# Auto-detect datasety.yaml in current directory
datasety workflow

# Specify file
datasety workflow --file pipeline.yaml

# Validate without running
datasety workflow --dry-run
```

## Options

| Option | Description | Default |
| --- | --- | --- |
| `--file`, `-f` | Path to workflow file | auto-detect |
| `--dry-run` | Validate without executing | `false` |

## File Format

Workflow files define a list of steps, each with a command and its arguments:

### YAML

```yaml
steps:
  - command: resize
    args:
      input: ./raw
      output: ./resized
      resolution: 768x1024
  - command: caption
    args:
      input: ./resized
      output: ./resized
      llm-api: true
      model: gpt-4o
```

### JSON

```json
{
  "steps": [
    {
      "command": "resize",
      "args": {
        "input": "./raw",
        "output": "./resized",
        "resolution": "768x1024"
      }
    }
  ]
}
```

## Argument Mapping

| YAML type | CLI equivalent |
| --- | --- |
| `key: value` | `--key value` |
| `key: true` | `--key` (flag) |
| `key: false` | (omitted) |
| `key: [a, b]` | `--key a --key b` |

## Auto-Detection

When no `--file` is specified, the workflow command searches for:

1. `datasety.yaml`
2. `datasety.yml`
3. `datasety.json`

## Dry Run

The `--dry-run` flag validates each step by:

1. Parsing arguments through the real argparse parser
2. Checking required parameters
3. Verifying input directories/files exist
4. Reporting pass/fail per step

No models are loaded and no images are processed.

## Examples

### LoRA Training Pipeline

```yaml
steps:
  - command: resize
    args:
      input: ./raw
      output: ./dataset
      resolution: 1024x1024
      crop-position: center
  - command: caption
    args:
      input: ./dataset
      output: ./dataset
      trigger-word: "[trigger]"
```

### Upscale Training Pipeline

```yaml
steps:
  - command: degrade
    args:
      input: ./originals
      output: ./dataset
      type:
        - random
      intensity-range: "0.2-0.8"
      paired: true
      seed: 42
  - command: align
    args:
      target: ./dataset/target
      control: ./dataset/control
  - command: caption
    args:
      input: ./dataset/target
      output: ./dataset/target
```
