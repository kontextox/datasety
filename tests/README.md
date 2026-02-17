# Tests

## Quick Start

```bash
# Run all CPU tests (default)
pytest

# Run with verbose output
pytest -v
```

## Test Structure

```
tests/
  test_resize.py          # calculate_resize_and_crop unit tests
  test_resize_cmd.py      # resize CLI integration tests
  test_caption.py         # caption CLI, LLM API, data URL tests
  test_align.py           # align CLI tests
  test_shuffle.py         # shuffle CLI, group parsing tests
  test_synthetic.py       # model family detection, pipeline kwargs, LoRA parsing
  test_mask.py            # mask CLI tests
  test_degrade.py         # degradation functions, pipeline, CLI tests
  test_character.py       # prompt parsing, LLM backend factory, CLI tests
  test_workflow.py        # workflow file discovery, parsing, dry-run, execution
  test_get_image_files.py # get_image_files utility tests
  test_gpu_models.py      # GPU-only model integration tests (marked @pytest.mark.gpu)
```

## GPU Tests

GPU tests are excluded by default (`addopts = "-m 'not gpu'"` in `pyproject.toml`).

### Overview

```
  Class                    Model                    Tests   Peak VRAM
  TestCaptionFlorence2     Florence-2-base          4       ~1 GB
  TestSyntheticQwen        Qwen-Image-Edit-2511     4       ~32 GB (sequential offload)
  TestSyntheticFluxKontext FLUX.1-Kontext-dev       2       ~33 GB (gated)
  TestSyntheticFlux2Klein  FLUX.2-klein-4B          2       ~8 GB
  TestSyntheticFlux2Dev    FLUX.2-dev               1       ~24 GB
  TestSyntheticLongCat     LongCat                  1       ~18 GB
  TestSyntheticSDXL        SDXL base 1.0            4       ~7 GB
  TestMaskCLIPSeg          CLIPSeg                  7       ~0.5 GB
  TestMaskGroundedSAM2     Grounding DINO + SAM 2   2       ~6 GB
  TestMaskSAM3             SAM 3                    2       ~5 GB (gated)
```

HunyuanImage is excluded (needs 48 GB).

### Prerequisites

```bash
pip install -e '.[all,dev]'
pip install git+https://github.com/huggingface/diffusers.git  # >= 0.37.0 for FLUX.2-klein

hf auth login

# Needed for gated models:
# https://huggingface.co/black-forest-labs/FLUX.2-dev
# https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
# https://huggingface.co/facebook/sam3
```

### Running GPU Tests

```bash
# All GPU tests
pytest -m gpu -v

# By family
pytest -m gpu -v -k caption
pytest -m gpu -v -k qwen
pytest -m gpu -v -k "flux_kontext"
pytest -m gpu -v -k "flux2_klein"
pytest -m gpu -v -k "flux2_dev"
pytest -m gpu -v -k longcat
pytest -m gpu -v -k sdxl
pytest -m gpu -v -k clipseg
pytest -m gpu -v -k "sam2"
pytest -m gpu -v -k sam3

# Skip gated models (no HF login needed)
pytest -m gpu -v -k "not (flux2_dev or flux_kontext or sam3)"
```

### Hardware Requirements

- **32 GB VRAM recommended** (RTX 5090, A100-40G) — handles all tests
- **24 GB works** (RTX 4090) — cpu-offload is auto-detected

## Troubleshooting

### Flux2KleinPipeline not found

- **Cause**: `diffusers < 0.37.0` doesn't include `Flux2KleinPipeline`
- **Fix**: `pip install git+https://github.com/huggingface/diffusers.git`

### Qwen OOM on 32 GB cards

- **Cause**: Qwen-Image-Edit uses ~31 GB peak VRAM
- **Fix**: Auto sequential offload triggers when free VRAM < needed

### Grounding DINO dtype mismatch

- **Cause**: processor outputs float32 tensors, model loaded in float16
- **Fix**: both Grounding DINO and SAM2 load in float32

### SAM2 post_process_masks KeyError

- **Cause**: `Sam2Processor` output doesn't always include `reshaped_input_sizes`
- **Fix**: threshold `pred_masks` directly and resize with PIL

### SAM3 wrong processor

- **Cause**: `AutoProcessor` resolves to `Sam3ImageProcessorFast` (no `text=` support)
- **Fix**: use `Sam3Processor` explicitly with `post_process_instance_segmentation`

---

# Full CLI Parameter Test Log (2026-02-16)

## Environment

- GPU: NVIDIA RTX 5090 (32 GB VRAM)
- RAM: 300 GB
- Python: 3.12, datasety 0.23.0
- API: OpenRouter (x-ai/grok-4.1-fast)
- Test images: `/workspace/images/{1,2}.jpg`

## 1. resize

| Parameter                          | Status | Notes                                               |
| ---------------------------------- | ------ | --------------------------------------------------- |
| `-i` / `-o` (batch)                | PASS   |                                                     |
| `--input-image` / `--output-image` | PASS   |                                                     |
| `-r` (resolution)                  | PASS   | Tested 512x512, 256x256, 768x1024, 400x600, 384x384 |
| `--crop-position top`              | PASS   |                                                     |
| `--crop-position center`           | PASS   |                                                     |
| `--crop-position bottom`           | PASS   |                                                     |
| `--crop-position left`             | PASS   |                                                     |
| `--crop-position right`            | PASS   |                                                     |
| `--input-format`                   | PASS   | Filtered to png only                                |
| `--output-format jpg`              | PASS   |                                                     |
| `--output-format png`              | PASS   |                                                     |
| `--output-format webp`             | PASS   |                                                     |
| `--output-name-numbers`            | PASS   | Files named 1.webp, 2.webp                          |

## 2. caption

| Parameter                            | Status | Notes                            |
| ------------------------------------ | ------ | -------------------------------- |
| `-i` / `-o` (batch)                  | PASS   |                                  |
| `--input-image` / `--output-caption` | PASS   |                                  |
| `--device cuda`                      | PASS   |                                  |
| `--device cpu`                       | PASS   | Florence-2-base on CPU           |
| `--device auto`                      | PASS   | Auto-detects CUDA                |
| `--trigger-word`                     | PASS   | "photo of sks person," prepended |
| `--prompt` (Florence-2)              | PASS   | `<MORE_DETAILED_CAPTION>`        |
| `--prompt` (LLM API)                 | PASS   | Custom prompt                    |
| `--model` (LLM API)                  | PASS   | x-ai/grok-4.1-fast               |
| `--num-beams`                        | PASS   | 5 beams tested                   |
| `--florence-2-base`                  | PASS   | Local GPU                        |
| `--florence-2-large`                 | PASS   | Local GPU                        |
| `--llm-api`                          | PASS   | OpenRouter                       |
| `--max-tokens`                       | PASS   | 100, 200                         |
| `--temperature`                      | PASS   | 0.3, 0.5                         |

## 3. align

| Parameter                   | Status | Notes                         |
| --------------------------- | ------ | ----------------------------- |
| `-t` / `-c`                 | PASS   |                               |
| `--multiple-of`             | PASS   | 64 tested, dimensions correct |
| `--output-format ""` (keep) | PASS   | Keeps original format         |
| `--output-format`           | PASS   | jpg conversion                |
| `--dry-run`                 | PASS   |                               |

## 4. shuffle

| Parameter             | Status | Notes                               |
| --------------------- | ------ | ----------------------------------- |
| `-i` / `-o`           | PASS   |                                     |
| `--group` (inline)    | PASS   |                                     |
| `--group` (file)      | PASS   |                                     |
| `--group` (URL)       | PASS   | Loaded 17 lines from GitHub raw URL |
| `--separator`         | PASS   | ", " tested                         |
| `--seed`              | PASS   | 42, 123, 7                          |
| `--dry-run`           | PASS   |                                     |
| `--show-distribution` | PASS   |                                     |

## 5. degrade

| Parameter                          | Status | Notes                  |
| ---------------------------------- | ------ | ---------------------- |
| `-i` / `-o`                        | PASS   |                        |
| `--input-image` / `--output-image` | PASS   |                        |
| `--type lowres`                    | PASS   |                        |
| `--type oversharpen`               | PASS   |                        |
| `--type noise`                     | PASS   |                        |
| `--type blur`                      | PASS   |                        |
| `--type jpeg`                      | PASS   |                        |
| `--type motion-blur`               | PASS   |                        |
| `--type pixelate`                  | PASS   |                        |
| `--type color-bands`               | PASS   |                        |
| `--type upscale-sim`               | PASS   |                        |
| `--type random`                    | PASS   |                        |
| `--intensity`                      | PASS   |                        |
| `--intensity-range`                | PASS   | 0.2-0.9, 0.3-0.7       |
| `--chain`                          | PASS   | Multiple types chained |
| `--num-variants`                   | PASS   | 3 variants tested      |
| `--paired`                         | PASS   | control/ + target/     |
| `--seed`                           | PASS   |                        |
| `--output-format png`              | PASS   |                        |
| `--output-format jpg`              | PASS   |                        |
| `--output-format webp`             | PASS   |                        |

## 6. mask

| Parameter                          | Status | Notes                                 |
| ---------------------------------- | ------ | ------------------------------------- |
| `-i` / `-o`                        | PASS   |                                       |
| `--input-image` / `--output-image` | PASS   |                                       |
| `-k` (keywords)                    | PASS   | face,hair,person                      |
| `--model sam3`                     | PASS   | Falls back to jetjodh/sam3            |
| `--model clipseg`                  | PASS   |                                       |
| `--model sam2`                     | PASS   | Grounding DINO + SAM2, 30.5% coverage |
| `--device cuda`                    | PASS   |                                       |
| `--threshold`                      | PASS   | 0.3, 0.4, 0.5                         |
| `--padding`                        | PASS   | 5, 10 px                              |
| `--blur`                           | PASS   | 3, 5 px                               |
| `--invert`                         | PASS   |                                       |
| `--naming folder`                  | PASS   |                                       |
| `--naming suffix`                  | PASS   | \_mask suffix                         |
| `--output-format png`              | PASS   |                                       |
| `--output-format jpg`              | PASS   |                                       |
| `--output-format webp`             | PASS   |                                       |
| `--dry-run`                        | PASS   |                                       |

## 7. synthetic

| Parameter                          | Status | Notes                                                                                                          |
| ---------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------- |
| `-i` / `-o`                        | PASS   |                                                                                                                |
| `--input-image` / `--output-image` | PASS   |                                                                                                                |
| `-p` (prompt)                      | PASS   |                                                                                                                |
| `--model`                          | PASS   | FLUX.2-klein-4B                                                                                                |
| `--weights`                        | PASS   | Code path verified: repo_id:filename, URL, local path parsing all work. Qwen-specific injection logic correct. |
| `--device cuda`                    | PASS   |                                                                                                                |
| `--device auto`                    | PASS   | Auto-detects CUDA                                                                                              |
| `--cpu-offload`                    | PASS   | Model CPU offload enabled with FLUX.2-klein                                                                    |
| `--steps`                          | PASS   | 4 steps                                                                                                        |
| `--cfg-scale`                      | PASS   | 2.0 tested                                                                                                     |
| `--true-cfg-scale`                 | PASS   | Qwen-specific, parsed correctly (not used with klein)                                                          |
| `--negative-prompt`                | PASS   | "blurry, ugly" tested                                                                                          |
| `--num-images`                     | PASS   | 2 images per input                                                                                             |
| `--seed`                           | PASS   |                                                                                                                |
| `--gguf`                           | PASS   | Code path verified: local paths, HF URLs, None all resolve correctly                                           |
| `--lora`                           | PASS   | Spec parsing works (path:weight). Full load pipeline works (needs peft). Tested with dummy safetensors.        |
| `--strength`                       | PASS   | 0.5, 0.7, 0.8 tested                                                                                           |
| `--output-format png`              | PASS   |                                                                                                                |
| `--output-format jpg`              | PASS   |                                                                                                                |
| `--output-format webp`             | PASS   |                                                                                                                |

## 8. character

| Parameter                 | Status | Notes                                                               |
| ------------------------- | ------ | ------------------------------------------------------------------- |
| `-r` (reference)          | PASS   | Single and multiple refs                                            |
| `-o` (output)             | PASS   |                                                                     |
| `-n` (num-images)         | PASS   | 3, 5                                                                |
| `--llm-api`               | PASS   | OpenRouter                                                          |
| `--character-description` | PASS   |                                                                     |
| `--style`                 | PASS   | "photorealistic", "anime illustration"                              |
| `--prompts-only`          | PASS   |                                                                     |
| `--prompts-file`          | PASS   |                                                                     |
| `--llm-ollama`            | PASS   | Code path verified, backend factory creates OllamaBackend correctly |
| `--llm-gguf`              | PASS   | Qwen2.5-0.5B-Instruct GGUF loaded, generated 3 prompts              |
| `--llm-model`             | PASS   | Qwen2.5-0.5B-Instruct HF model loaded, generated 2 prompts          |
| `--seed`                  | PASS   | Parsed correctly                                                    |
| `--steps`                 | PASS   | Parsed correctly                                                    |
| `--cfg-scale`             | PASS   | Parsed correctly                                                    |
| `--output-format jpg`     | PASS   | Parsed correctly                                                    |
| Full generation           | N/T    | Requires FLUX.2-klein-4B (8GB) download                             |

## 9. workflow

| Parameter       | Status | Notes                  |
| --------------- | ------ | ---------------------- |
| `-f` YAML       | PASS   |                        |
| `-f` JSON       | PASS   |                        |
| auto-detect     | PASS   |                        |
| `--dry-run`     | PASS   |                        |
| multi-step exec | PASS   | resize→caption→degrade |

## Summary

- **Unit tests**: 202/202 PASS (27 GPU-only deselected by default)
- **Commands tested**: 9/9 (resize, caption, align, shuffle, degrade, mask, synthetic, character, workflow)
- **Parameters tested**: 100+ unique parameter combinations
- **Models tested**: Florence-2-base, Florence-2-large, CLIPSeg, SAM3, SAM2, FLUX.2-klein-4B, Qwen2.5-0.5B-Instruct (GGUF + HF)
- **APIs tested**: OpenRouter (x-ai/grok-4.1-fast) via LLM API
- **LLM backends**: OpenAI API, GGUF (llama-cpp-python), HuggingFace transformers, Ollama (factory verified)
