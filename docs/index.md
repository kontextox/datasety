---
layout: home

hero:
  name: ~# datasety
  text: Dataset - is easy!
  tagline: One tool for the full dataset pipeline — resize, caption, align, generate, mask, degrade, and automate with workflows.
  image:
    src: /my-hero-image.png 
    alt: Full dataset pipeline
  actions:
    - theme: brand
      text: Get Started
      link: /getting-started
    - theme: alt
      text: GitHub
      link: https://github.com/kontextox/datasety

features:
  - icon: "&#x1F4D0;"
    title: Resize & Crop
    details: Batch resize to exact dimensions with top/center/bottom crop. Supports single-image and directory modes.
    link: /commands/resize
    linkText: Learn more
  - icon: "&#x1F4DD;"
    title: Caption
    details: Florence-2 (local, 0.23B or 0.77B) or any OpenAI-compatible vision API. Trigger words, custom prompts.
    link: /commands/caption
    linkText: Learn more
  - icon: "&#x1F3A8;"
    title: Synthetic Editing
    details: Image editing with FLUX.2, Qwen, SDXL, LongCat, HunyuanImage. LoRA, GGUF quantization, CPU offload.
    link: /commands/synthetic
    linkText: Learn more
  - icon: "&#x1F3AD;"
    title: Segmentation Masks
    details: Text-prompted masks with SAM 3, Grounded SAM 2, or CLIPSeg. Padding, blur, invert options.
    link: /commands/mask
    linkText: Learn more
  - icon: "&#x1F9D1;"
    title: Character Generation
    details: Identity-preserving datasets from reference faces. LLM-generated prompts + IP-Adapter.
    link: /commands/character
    linkText: Learn more
  - icon: "&#x1F504;"
    title: Workflows
    details: Define multi-step pipelines in YAML/JSON. Dry-run validates everything before execution.
    link: /commands/workflow
    linkText: Learn more
  - icon: "&#x1F517;"
    title: Align Pairs
    details: Match control/target dimensions, enforce multiples of 32, unify formats for training pairs.
    link: /commands/align
    linkText: Learn more
  - icon: "&#x1F4C9;"
    title: Degrade
    details: 9 degradation types for upscale training — JPEG artifacts, noise, blur, pixelation, and more. Pure Pillow.
    link: /commands/degrade
    linkText: Learn more
  - icon: "&#x1F3B2;"
    title: Shuffle Captions
    details: Random caption generation from text groups. Inline, file, or URL sources with seed control.
    link: /commands/shuffle
    linkText: Learn more
---

## Quick Install

```bash
pip install datasety          # core
pip install datasety[all]     # everything
```

## Example Pipeline

```bash
# 1. Resize raw photos
datasety resize -i ./raw -o ./dataset -r 1024x1024

# 2. Generate captions with a trigger word
datasety caption -i ./dataset -o ./dataset --trigger-word "[trigger]"

# 3. Generate face masks for focused training
datasety mask -i ./dataset -o ./masks -k "face,hair"
```

Or define it as a workflow:

```yaml
# datasety.yaml
steps:
  - command: resize
    args: { input: ./raw, output: ./dataset, resolution: 1024x1024 }
  - command: caption
    args: { input: ./dataset, output: ./dataset, trigger-word: "[trigger]" }
  - command: mask
    args: { input: ./dataset, output: ./masks, keywords: "face,hair" }
```

```bash
datasety workflow --dry-run    # validate
datasety workflow              # execute
```
