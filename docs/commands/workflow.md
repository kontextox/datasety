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

| Option         | Description                | Default     |
| -------------- | -------------------------- | ----------- |
| `--file`, `-f` | Path to workflow file      | auto-detect |
| `--dry-run`    | Validate without executing | `false`     |

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
      model: gpt-5-nano
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

| YAML type     | CLI equivalent    |
| ------------- | ----------------- |
| `key: value`  | `--key value`     |
| `key: true`   | `--key` (flag)    |
| `key: false`  | (omitted)         |
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

### Face LoRA with Masks

The most common pipeline: resize raw photos, caption with a trigger word, and generate face masks for focused training loss.

```yaml
steps:
  - command: resize
    args:
      input: ./raw
      output: ./dataset
      resolution: 1024x1024
      crop-position: top
  - command: caption
    args:
      input: ./dataset
      output: ./dataset
      trigger-word: "ohwx person,"
  - command: mask
    args:
      input: ./dataset
      output: ./dataset/masks
      keywords: "person,face,hair"
      model: clipseg
      threshold: 0.4
      padding: 10
      blur: 5
```

### Synthetic Augmentation + Re-caption

Expand a small dataset with edited variations, then caption the results.

```yaml
steps:
  - command: synthetic
    args:
      input: ./dataset
      output: ./augmented
      prompt: "the person is wearing a knitted beanie hat"
      steps: 4
      cfg-scale: 2.5
      seed: 42
  - command: caption
    args:
      input: ./augmented
      output: ./augmented
      trigger-word: "ohwx person,"
```

### Upscale Training (Paired Degradation)

Create a paired dataset for super-resolution training. Chains JPEG, noise, and blur artifacts.

```yaml
steps:
  - command: resize
    args:
      input: ./originals
      output: ./resized
      resolution: 1024x1024
  - command: degrade
    args:
      input: ./resized
      output: ./dataset
      type:
        - jpeg
        - noise
        - blur
      chain: true
      intensity-range: "0.3-0.7"
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

### Inpainting Dataset

Resize, generate masks for removable accessories, and caption.

```yaml
steps:
  - command: resize
    args:
      input: ./photos
      output: ./dataset
      resolution: 1024x1024
      crop-position: top
  - command: mask
    args:
      input: ./dataset
      output: ./dataset/masks
      keywords: "hat,glasses,sunglasses,scarf,necklace"
      model: sam3
      threshold: 0.3
      padding: 5
      blur: 3
  - command: caption
    args:
      input: ./dataset
      output: ./dataset
      florence-2-large: true
```

See [Workflows](/workflows) for more real-world pipelines including background replacement, product LoRAs, multi-resolution datasets, and sweep-then-train patterns.
