# GPU Tests

## Overview

```bash
  ┌──────────────────────────┬────────────────────────┬───────┬─────────────────────────────┐
  │          Class           │         Model          │ Tests │          Peak VRAM          │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestCaptionFlorence2     │ Florence-2-base        │ 4     │ ~1 GB                       │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticQwen        │ Qwen-Image-Edit-2511   │ 4     │ ~32 GB (sequential offload) │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticFluxKontext │ FLUX.1-Kontext-dev     │ 2     │ ~33 GB (gated)              │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticFlux2Klein  │ FLUX.2-klein-4B        │ 2     │ ~8 GB                       │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticFlux2Dev    │ FLUX.2-dev             │ 1     │ ~24 GB                      │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticLongCat     │ LongCat                │ 1     │ ~18 GB                      │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticSDXL        │ SDXL base 1.0          │ 4     │ ~7 GB                       │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestMaskCLIPSeg          │ CLIPSeg                │ 7     │ ~0.5 GB                     │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestMaskGroundedSAM2     │ Grounding DINO + SAM 2 │ 2     │ ~6 GB                       │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestMaskSAM3             │ SAM 3                  │ 2     │ ~5 GB (gated)               │
  └──────────────────────────┴────────────────────────┴───────┴─────────────────────────────┘
```

HunyuanImage is excluded (needs 48 GB).

## Prerequisites

```bash
pip install -e '.[all,dev]'
hf auth login          # needed for gated models (FLUX Kontext, SAM 3)
pip install git+https://github.com/huggingface/diffusers.git  # diffusers >= 0.37.0 for FLUX.2-klein
```

## Running tests

### All GPU tests

```bash
pytest -m gpu -v
```

### By family

```bash
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
```

### Skip gated models (no HF login needed)

```bash
pytest -m gpu -v -k "not (flux2_dev or flux_kontext or sam3)"
```

## Hardware requirements

- **32 GB VRAM recommended** (RTX 5090, A100-40G) — handles all tests comfortably
- **24 GB works** (RTX 4090) — cpu-offload is auto-detected based on available VRAM

## Troubleshooting: known issues and fixes

### Flux2KleinPipeline not found

- **Cause**: `diffusers < 0.37.0` doesn't include `Flux2KleinPipeline`
- **Fix**: `pip install git+https://github.com/huggingface/diffusers.git`
- The pipeline auto-upgrades on first use, but pre-installing avoids the delay

### Qwen OOM on 32 GB cards

- **Cause**: Qwen-Image-Edit uses ~31 GB peak VRAM
- **Fix**: `_MODEL_VRAM_GB["qwen"] = 32` triggers auto sequential offload
- Sequential vs model offload: sequential moves individual layers to CPU/GPU on demand, model offload moves whole submodels

### Grounding DINO dtype mismatch (float vs half)

- **Cause**: processor outputs float32 tensors, model loaded in float16
- **Fix**: load both Grounding DINO and SAM2 in float32 (extra ~1.4 GB VRAM, negligible)

### SAM2 post_process_masks KeyError

- **Cause**: `Sam2Processor` output doesn't always include `reshaped_input_sizes`
- **Fix**: threshold `pred_masks` directly and resize with PIL, skip `post_process_masks()`

### SAM3 wrong processor

- **Cause**: `AutoProcessor` resolves to `Sam3ImageProcessorFast` (no `text=` support)
- **Fix**: use `Sam3Processor` explicitly, and `post_process_instance_segmentation` (not `post_process_masks`)
