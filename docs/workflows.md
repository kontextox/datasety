# Workflows

Workflows let you define multi-step datasety pipelines in YAML or JSON files. This is useful for reproducible dataset preparation.

## Quick Start

Create `datasety.yaml` in your project directory:

```yaml
steps:
  - command: resize
    args:
      input: ./raw
      output: ./dataset
      resolution: 1024x1024
  - command: caption
    args:
      input: ./dataset
      output: ./dataset
      trigger-word: "[trigger]"
```

Validate first, then run:

```bash
datasety workflow --dry-run
datasety workflow
```

## File Format

See the [workflow command reference](/commands/workflow) for full format details.

## Common Pipelines

### LoRA Training Dataset

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

### Control/Target LoRA (e.g., inpainting)

```yaml
steps:
  - command: align
    args:
      target: ./target
      control: ./control
  - command: caption
    args:
      input: ./target
      output: ./target
  - command: mask
    args:
      input: ./target
      output: ./masks
      keywords: "face,hair"
```

### Upscale Training

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

### Synthetic Augmentation

```yaml
steps:
  - command: resize
    args:
      input: ./raw
      output: ./dataset
      resolution: 768x1024
  - command: synthetic
    args:
      input: ./dataset
      output: ./augmented
      prompt: "add a winter hat"
      steps: 4
  - command: caption
    args:
      input: ./augmented
      output: ./augmented
      llm-api: true
      model: gpt-4o
```

### Character Dataset

```yaml
steps:
  - command: character
    args:
      reference:
        - face.jpg
      output: ./character_raw
      llm-ollama: llama3.2
      num-images: 20
      style: photorealistic
      prompts-only: true
```
