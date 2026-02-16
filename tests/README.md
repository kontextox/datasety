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
pytest -m gpu -v -k "grounded_sam2"
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
