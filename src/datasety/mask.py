"""Generate binary masks from images using text keywords."""

import sys
from pathlib import Path

from PIL import Image

from datasety.common import _resolve_io_mode, get_image_files, resolve_device


def _load_mask_model_sam3(device, torch_dtype):
    """Load SAM 3 for text-prompted segmentation."""
    from transformers import Sam3Model, Sam3Processor

    primary = "facebook/sam3"
    fallback = "jetjodh/sam3"
    try:
        processor = Sam3Processor.from_pretrained(primary)
        model = (
            Sam3Model.from_pretrained(
                primary,
                torch_dtype=torch_dtype,
            )
            .to(device)
            .eval()
        )
        return model, processor
    except Exception as e:
        print(f"Could not load {primary} ({e}), falling back to {fallback}")
    processor = Sam3Processor.from_pretrained(fallback)
    model = (
        Sam3Model.from_pretrained(
            fallback,
            torch_dtype=torch_dtype,
        )
        .to(device)
        .eval()
    )
    return model, processor


def _load_mask_model_grounded_sam2(device, torch_dtype):
    """Load Grounding DINO + SAM 2 for grounded segmentation.

    Note: torch_dtype is accepted for API consistency but both models are
    loaded in float32 to avoid dtype mismatches with processor outputs.
    """
    from transformers import (
        AutoModelForZeroShotObjectDetection,
        AutoProcessor,
        Sam2Model,
        Sam2Processor,
    )

    # Load both models in float32: the processor outputs float32 tensors and
    # mixed-precision (float16 model + float32 inputs) causes dtype errors in
    # Grounding DINO's cross-attention layers.
    dino_id = "IDEA-Research/grounding-dino-base"
    dino_processor = AutoProcessor.from_pretrained(dino_id)
    dino_model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(
            dino_id,
        )
        .to(device)
        .eval()
    )

    sam2_id = "facebook/sam2-hiera-large"
    sam2_processor = Sam2Processor.from_pretrained(sam2_id)
    sam2_model = (
        Sam2Model.from_pretrained(
            sam2_id,
        )
        .to(device)
        .eval()
    )
    return (dino_model, dino_processor, sam2_model, sam2_processor)


def _load_mask_model_clipseg(device, torch_dtype):
    """Load CLIPSeg for text-based segmentation."""
    from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = (
        CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined",
        )
        .to(device)
        .eval()
    )
    return model, processor


def _segment_sam3(model, processor, image, keywords, threshold, device):
    """Run SAM 3 segmentation for each keyword and return combined mask."""
    import numpy as np
    import torch

    w, h = image.size
    combined = np.zeros((h, w), dtype=np.uint8)

    for keyword in keywords:
        inputs = processor(images=image, text=keyword, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            target_sizes=inputs["original_sizes"].tolist(),
        )[0]
        for mask in results["masks"]:
            m = mask.cpu().numpy()
            combined = np.maximum(combined, m.astype(np.uint8) * 255)

    return combined


def _segment_grounded_sam2(models, image, keywords, threshold, device):
    """Run Grounding DINO + SAM 2 segmentation."""
    import numpy as np
    import torch

    dino_model, dino_processor, sam2_model, sam2_processor = models
    w, h = image.size
    combined = np.zeros((h, w), dtype=np.uint8)

    # Grounding DINO: detect boxes for all keywords at once
    text = ". ".join(keywords) + "."
    inputs = dino_processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)

    results = dino_processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu()
    if len(boxes) == 0:
        return combined

    # SAM 2: segment within each detected box (single best mask per box)
    sam2_inputs = sam2_processor(
        images=image,
        input_boxes=[boxes.tolist()],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        sam2_outputs = sam2_model(**sam2_inputs, multimask_output=False)

    # Use processor's post_process_masks for proper bilinear upsampling
    # (avoids blocky nearest-neighbor artifacts from raw low-res pred_masks)
    try:
        masks = sam2_processor.post_process_masks(
            sam2_outputs.pred_masks.cpu(),
            sam2_inputs["original_sizes"],
            binarize=True,
        )
        for mask_tensor in masks:
            for obj_mask in mask_tensor:
                m = obj_mask.squeeze().numpy().astype(np.uint8) * 255
                if m.shape != (h, w):
                    m = np.array(Image.fromarray(m).resize((w, h), Image.BILINEAR))
                combined = np.maximum(combined, m)
    except (TypeError, KeyError):
        # Fallback for older transformers that lack post_process_masks support
        pred_masks = sam2_outputs.pred_masks.cpu().numpy()
        flat = pred_masks.reshape(-1, pred_masks.shape[-2], pred_masks.shape[-1])
        for m in flat:
            # Upsample continuous logits with bilinear first, then threshold
            m_img = Image.fromarray(m).resize((w, h), Image.BILINEAR)
            m = (np.array(m_img) > 0).astype(np.uint8) * 255
            combined = np.maximum(combined, m)

    return combined


def _segment_clipseg(model, processor, image, keywords, threshold, device):
    """Run CLIPSeg segmentation for each keyword."""
    import numpy as np
    import torch

    w, h = image.size
    combined = np.zeros((h, w), dtype=np.float32)

    for keyword in keywords:
        inputs = processor(
            text=[keyword],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits[0]  # (H_clip, W_clip)
        probs = torch.sigmoid(logits).cpu().numpy()
        # Resize to full resolution
        prob_img = Image.fromarray(probs).resize((w, h), Image.BILINEAR)
        combined = np.maximum(combined, np.array(prob_img))

    return ((combined >= threshold) * 255).astype(np.uint8)


def cmd_mask(args):
    """Generate binary masks from images using text keywords."""
    try:
        import numpy as np
        import torch
    except ImportError:
        print("Error: Required packages not installed.")
        print("Run: pip install 'datasety[mask]'")
        sys.exit(1)

    single_files, output_path_resolved, is_single = _resolve_io_mode(args)

    if is_single:
        image_files = single_files
        output_dir = output_path_resolved.parent if output_path_resolved else Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        input_dir = Path(args.input)
        output_dir = Path(args.output) if args.naming == "folder" else input_dir

        if not input_dir.exists():
            print(f"Error: Input directory '{input_dir}' does not exist.")
            sys.exit(1)

        if args.naming == "folder":
            output_dir.mkdir(parents=True, exist_ok=True)

    # Determine device
    device = resolve_device(args.device)

    torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    # Parse keywords
    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    if not keywords:
        print("Error: No valid keywords provided.")
        sys.exit(1)

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Found {len(keywords)} keywords: {keywords}")
    print(f"Threshold: {args.threshold}")
    if args.dry_run:
        print("=== DRY RUN (no files will be saved) ===")
    print("-" * 50)

    # Load model
    print("Loading segmentation model...")
    try:
        if args.model == "sam3":
            models = _load_mask_model_sam3(device, torch_dtype)
        elif args.model == "sam2":
            models = _load_mask_model_grounded_sam2(device, torch_dtype)
        elif args.model == "clipseg":
            models = _load_mask_model_clipseg(device, torch_dtype)
        else:
            print(f"Error: Unknown model '{args.model}'. Use: sam3, sam2, clipseg")
            sys.exit(1)
    except ImportError as e:
        print(f"Error: Missing dependency: {e}")
        print("Run: pip install 'datasety[mask]'")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    # Find images
    if not is_single:
        formats = ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]
        image_files = get_image_files(Path(args.input), formats, recursive=args.recursive)

        if not image_files:
            print(f"No images found in '{args.input}'")
            sys.exit(0)

    print(f"Found {len(image_files)} images")
    print("-" * 50)

    processed = 0
    total = len(image_files)
    out_fmt = args.output_format

    for idx, img_path in enumerate(image_files, 1):
        try:
            with Image.open(img_path) as img:
                image = img.convert("RGB")
                w, h = image.size

                # Run segmentation
                if args.model == "sam3":
                    mask_array = _segment_sam3(
                        models[0],
                        models[1],
                        image,
                        keywords,
                        args.threshold,
                        device,
                    )
                elif args.model == "sam2":
                    mask_array = _segment_grounded_sam2(
                        models,
                        image,
                        keywords,
                        args.threshold,
                        device,
                    )
                elif args.model == "clipseg":
                    mask_array = _segment_clipseg(
                        models[0],
                        models[1],
                        image,
                        keywords,
                        args.threshold,
                        device,
                    )

            # Apply padding (dilation)
            if args.padding > 0:
                from PIL import ImageFilter

                mask_img = Image.fromarray(mask_array, mode="L")
                mask_img = mask_img.filter(ImageFilter.MaxFilter(size=args.padding * 2 + 1))
                mask_array = np.array(mask_img)

            # Apply blur
            if args.blur > 0:
                from PIL import ImageFilter

                mask_img = Image.fromarray(mask_array, mode="L")
                mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=args.blur))
                mask_array = np.array(mask_img)

            # Invert if requested
            if args.invert:
                mask_array = 255 - mask_array

            # Determine output path
            if is_single and output_path_resolved:
                out_path = output_path_resolved
            elif args.naming == "folder":
                out_path = output_dir / f"{img_path.stem}.{out_fmt}"
            else:
                out_path = Path(args.input) / f"{img_path.stem}_mask.{out_fmt}"

            pixel_count = int(np.sum(mask_array > 127))
            coverage = pixel_count / (w * h) * 100

            if args.dry_run:
                print(
                    f"  [{idx}/{total}] {img_path.name}: {len(keywords)} keywords, "
                    f"{coverage:.1f}% coverage -> {out_path.name}"
                )
            else:
                mask_img = Image.fromarray(mask_array, mode="L")
                mask_img.save(out_path)
                print(
                    f"[{idx}/{total}] [OK] {img_path.name} -> "
                    f"{out_path.name} ({coverage:.1f}% coverage)"
                )

            processed += 1

        except Exception as e:
            print(f"[{idx}/{total}] [ERROR] {img_path.name}: {e}")

    print("-" * 50)
    print(f"Done! Processed: {processed} images")


def register_parser(subparsers):
    """Register the mask subcommand."""
    mask_parser = subparsers.add_parser(
        "mask", help="Generate binary masks from images using text keywords"
    )
    mask_parser.add_argument("--input", "-i", default="", help="Input directory containing images")
    mask_parser.add_argument("--output", "-o", default="", help="Output directory for mask images")
    mask_parser.add_argument(
        "--input-image", default=None, help="Single input image path (alternative to --input dir)"
    )
    mask_parser.add_argument(
        "--output-image", default=None, help="Single output mask path (use with --input-image)"
    )
    mask_parser.add_argument(
        "--keywords",
        "-k",
        required=True,
        help="Comma-separated keywords to segment (e.g., 'face,hair,hat')",
    )
    mask_parser.add_argument(
        "--model",
        choices=["sam3", "sam2", "clipseg"],
        default="sam3",
        help="Segmentation model (default: sam3)",
    )
    mask_parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda", "mps"],
        default="auto",
        help="Device to run model on (default: auto-detect GPU/MPS)",
    )
    mask_parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence threshold for detection (0.0-1.0, default: 0.3)",
    )
    mask_parser.add_argument(
        "--padding", type=int, default=0, help="Pixels to expand mask by (dilation, default: 0)"
    )
    mask_parser.add_argument(
        "--blur",
        type=int,
        default=0,
        help="Gaussian blur radius for mask edges (0=sharp, default: 0)",
    )
    mask_parser.add_argument(
        "--invert", action="store_true", help="Invert mask (black=ROI, white=ignore)"
    )
    mask_parser.add_argument(
        "--naming",
        choices=["folder", "suffix"],
        default="folder",
        help="Output naming: 'folder' (same name in output dir) or "
        "'suffix' (_mask suffix in input dir) (default: folder)",
    )
    mask_parser.add_argument(
        "--output-format",
        choices=["png", "jpg", "webp"],
        default="png",
        help="Output image format (default: png)",
    )
    mask_parser.add_argument(
        "--dry-run", action="store_true", help="Preview detections without saving masks"
    )
    mask_parser.add_argument(
        "--recursive",
        "-R",
        action="store_true",
        help="Search input directory recursively for images",
    )
    mask_parser.set_defaults(func=cmd_mask)
