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
      trigger-word: "ohwx,"
```

Validate first, then run:

```bash
datasety workflow --dry-run
datasety workflow
```

## File Format

See the [workflow command reference](/commands/workflow) for full format details.

## Real-World Pipelines

### Face/Person LoRA Training

The most common use case: prepare a face LoRA dataset from raw selfies or portrait photos. Resize to square, caption with a rare trigger word, and generate face masks so the trainer can focus loss on the subject.

```yaml
# face-lora.yaml
# Input: ./raw/ containing 15-30 portrait photos (JPG/PNG from phone camera)
# Output: ./dataset/ with resized images, captions (.txt), and masks
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

```bash
datasety workflow -f face-lora.yaml --dry-run
datasety workflow -f face-lora.yaml
# Result: ./dataset/ has 001.jpg + 001.txt + masks/001.png for each image
```

### Accessory Augmentation

You have 20 photos of a person and want to expand the dataset with synthetic variations wearing different accessories. This is useful when you want the LoRA to generalize beyond the reference photos.

```yaml
# augment-accessories.yaml
# Input: ./dataset/ containing resized training images (from face LoRA step above)
# Output: ./augmented/ with synthetic edits, then re-captioned
steps:
  - command: synthetic
    args:
      input: ./dataset
      output: ./augmented/hats
      prompt: "the person is wearing a knitted beanie hat"
      steps: 4
      cfg-scale: 2.5
      seed: 42

  - command: synthetic
    args:
      input: ./dataset
      output: ./augmented/glasses
      prompt: "the person is wearing round sunglasses"
      steps: 4
      cfg-scale: 2.5
      seed: 42

  - command: synthetic
    args:
      input: ./dataset
      output: ./augmented/scarves
      prompt: "the person is wearing a red wool scarf"
      steps: 4
      cfg-scale: 2.5
      seed: 42

  - command: caption
    args:
      input: ./augmented/hats
      output: ./augmented/hats
      trigger-word: "ohwx person,"

  - command: caption
    args:
      input: ./augmented/glasses
      output: ./augmented/glasses
      trigger-word: "ohwx person,"

  - command: caption
    args:
      input: ./augmented/scarves
      output: ./augmented/scarves
      trigger-word: "ohwx person,"
```

### Product Photography LoRA

Prepare a dataset from product photos for an object LoRA. Products often have white or cluttered backgrounds, so we use non-square portrait crops to preserve product shape.

```yaml
# product-lora.yaml
# Input: ./product_photos/ containing product images (various sizes)
# Output: ./dataset/ ready for training
steps:
  - command: resize
    args:
      input: ./product_photos
      output: ./dataset
      resolution: 768x1024
      crop-position: center

  - command: caption
    args:
      input: ./dataset
      output: ./dataset
      trigger-word: "sks product,"
      florence-2-large: true

  - command: mask
    args:
      input: ./dataset
      output: ./dataset/masks
      keywords: "product,object,item"
      model: clipseg
      threshold: 0.3
```

### Upscale/Restore Training

Create a paired dataset for training an upscale or image restoration model. The degradation step creates realistic artifacts (JPEG compression, noise, blur) that the model learns to reverse.

```yaml
# upscale-training.yaml
# Input: ./originals/ containing high-quality source images
# Output: ./dataset/ with control/ (degraded) and target/ (original) subdirs
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

### Background Replacement

Generate inverted masks (everything except the subject), then use synthetic editing to change backgrounds. Useful for placing subjects in varied environments.

```yaml
# background-swap.yaml
# Input: ./portraits/ containing people photos with plain backgrounds
# Output: Three sets of re-backgrounded images
steps:
  - command: resize
    args:
      input: ./portraits
      output: ./resized
      resolution: 1024x1024
      crop-position: center

  - command: synthetic
    args:
      input: ./resized
      output: ./bg_outdoor
      prompt: "the person is standing in a sunny park with trees and grass"
      steps: 4
      cfg-scale: 2.5
      seed: 100

  - command: synthetic
    args:
      input: ./resized
      output: ./bg_studio
      prompt: "professional studio portrait with soft lighting and gray backdrop"
      steps: 4
      cfg-scale: 2.5
      seed: 100

  - command: synthetic
    args:
      input: ./resized
      output: ./bg_urban
      prompt: "the person is standing on a city street with buildings"
      steps: 4
      cfg-scale: 2.5
      seed: 100
```

### Inpainting Dataset

Create an inpainting training dataset with source images, masks, and captions. The masks mark regions to inpaint (e.g., accessories that should be removable).

```yaml
# inpainting-dataset.yaml
# Input: ./photos/ containing images of people with accessories
# Output: ./dataset/ with images, masks for accessories, and captions
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
      keywords: "hat,glasses,sunglasses,scarf,necklace,earring"
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

### Vision API Captioning with Custom Provider

Use a third-party OpenAI-compatible API for captioning when you want higher-quality descriptions than Florence-2. Works with OpenRouter, Together, or any compatible endpoint.

```yaml
# api-caption.yaml
# Requires: OPENAI_API_KEY and OPENAI_BASE_URL env vars
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
      llm-api: true
      model: gpt-4o
      trigger-word: "ohwx person,"
      prompt: "Describe this person's appearance, clothing, pose, expression, and setting in one detailed paragraph. Do not mention image quality or photography terms."
      temperature: 0.3
      max-tokens: 200
```

```bash
# Run with OpenRouter
OPENAI_BASE_URL=https://openrouter.ai/api/v1 \
OPENAI_API_KEY=sk-or-... \
datasety workflow -f api-caption.yaml
```

### Multi-Resolution Dataset

Some trainers benefit from images at multiple resolutions. This workflow outputs the same source at three common training sizes.

```yaml
# multi-res.yaml
# Input: ./raw/ containing high-res source images (>= 2048px)
steps:
  - command: resize
    args:
      input: ./raw
      output: ./dataset_512
      resolution: 512x512

  - command: resize
    args:
      input: ./raw
      output: ./dataset_768
      resolution: 768x768

  - command: resize
    args:
      input: ./raw
      output: ./dataset_1024
      resolution: 1024x1024

  - command: caption
    args:
      input: ./dataset_1024
      output: ./dataset_512
      trigger-word: "ohwx,"

  - command: caption
    args:
      input: ./dataset_1024
      output: ./dataset_768
      trigger-word: "ohwx,"

  - command: caption
    args:
      input: ./dataset_1024
      output: ./dataset_1024
      trigger-word: "ohwx,"
```

### Sweep Then Train

Use `sweep` to find optimal generation parameters on a small sample, then apply the best settings to the full dataset.

```bash
# Step 1: Test on 2-3 images to find the best steps + cfg-scale
mkdir ./sample && cp ./dataset/001.jpg ./dataset/002.jpg ./sample/

datasety sweep \
    -i ./sample -o ./sweep_results \
    -p "the person is wearing aviator sunglasses" \
    --steps 2,4,8 \
    --cfg-scale 1.5,2.5,3.5 \
    --seed 42 --run

# Step 2: Visually inspect ./sweep_results/steps4_cfg2.5/ etc.
# Pick the best combination, then apply to the full dataset:
```

```yaml
# full-augment.yaml
steps:
  - command: synthetic
    args:
      input: ./dataset
      output: ./augmented
      prompt: "the person is wearing aviator sunglasses"
      steps: 4
      cfg-scale: 2.5
      seed: 42

  - command: caption
    args:
      input: ./augmented
      output: ./augmented
      trigger-word: "ohwx person,"
```

### Shuffled Caption Augmentation

Generate randomized captions to add variety to a training dataset. Each image gets a randomly assembled caption from predefined text groups, which helps prevent the model from memorizing exact phrasings.

```yaml
# shuffle-captions.yaml
# Input: ./raw/ containing images
# Generates randomized captions from text groups
steps:
  - command: resize
    args:
      input: ./raw
      output: ./dataset
      resolution: 1024x1024

  - command: shuffle
    args:
      input: ./dataset
      output: ./dataset
      group:
        - "ohwx person,|a photo of ohwx,|ohwx,"
        - "looking at the camera|facing forward|in a relaxed pose|smiling"
        - "natural lighting|soft studio light|bright daylight|warm indoor lighting"
      seed: 42
```
