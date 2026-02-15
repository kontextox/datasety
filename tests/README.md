```bash
  ┌──────────────────────────┬────────────────────────┬───────┬─────────────────────────────┐
  │          Class           │         Model          │ Tests │          Peak VRAM          │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestCaptionFlorence2     │ Florence-2-base        │ 4     │ ~1 GB                       │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticQwen        │ Qwen-Image-Edit-2511   │ 4     │ ~16 GB                      │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticFluxKontext │ FLUX.1-Kontext-dev     │ 2     │ ~24 GB (gated)              │
  ├──────────────────────────┼────────────────────────┼───────┼─────────────────────────────┤
  │ TestSyntheticFlux2Klein  │ FLUX.2-klein-4B        │ 2     │ ~8 GB                       │
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

How to run on your server

```bash
pip install -e '.[all,dev]'
hf auth login # needed for FLUX Kontext + SAM 3
```

# All GPU tests

```bash
pytest -m gpu -v
```

# By family

```bash
pytest -m gpu -v -k caption
pytest -m gpu -v -k qwen
pytest -m gpu -v -k "flux_kontext"
pytest -m gpu -v -k "flux2_klein"
pytest -m gpu -v -k sdxl
pytest -m gpu -v -k clipseg
pytest -m gpu -v -k "grounded_sam2"
pytest -m gpu -v -k sam3
```

# Skip gated models (no HF login needed)

```bash
pytest -m gpu -v -k "not (flux_kontext or sam3)"
```

> The RTX 5090 (32 GB) at $0.3/hr handles all of these comfortably.
> An RTX 4090 (24 GB) works too — cpu-offload is auto-detected
> based on available VRAM.
