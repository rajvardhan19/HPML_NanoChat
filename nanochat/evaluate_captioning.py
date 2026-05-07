"""
nanochat/vision/evaluate_captioning.py

Automatic caption quality evaluation using BLEU and CIDEr metrics.

What this script does
---------------------
1. Load a trained NanoChatVisionModel checkpoint.
2. Run model.generate() on every image in the COCO val2017 set.
3. Collect generated captions vs. reference captions.
4. Compute BLEU-1/2/3/4 (NLTK) and CIDEr (pycocotools / pycocoevalcap).
5. Save results to results/eval_results.json.

Why BLEU + CIDEr?
-----------------
  BLEU (Bilingual Evaluation Understudy): n-gram precision with brevity penalty.
    - Fast, interpretable, widely used in machine translation and captioning.
    - BLEU-4 is the most cited; BLEU-1 measures unigram coverage.

  CIDEr (Consensus-based Image Description Evaluation): TF-IDF weighted n-gram
    similarity that accounts for caption consensus across multiple references.
    - More robust than BLEU for captioning because it down-weights common phrases
      (e.g. "a man") that are not discriminative across images.
    - Standard metric on COCO captioning leaderboards.

Usage
-----
# Evaluate a checkpoint on the full COCO val set:
python -m nanochat.vision.evaluate_captioning \
    --checkpoint checkpoints/opt_all/model.pt \
    --config checkpoints/opt_all/config.json \
    --data_root data/coco \
    --output_dir results

# Quick smoke test (first 100 images, greedy decode):
python -m nanochat.vision.evaluate_captioning \
    --checkpoint checkpoints/opt_all/model.pt \
    --config checkpoints/opt_all/config.json \
    --data_root data/coco \
    --max_val_samples 100 \
    --temperature 0.0

Dependencies
------------
  pip install nltk pycocoevalcap pycocotools
  python -c "import nltk; nltk.download('punkt')"
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

import torch

from nanochat.vision.config import VisionExperimentConfig, VisionModelConfig
from nanochat.vision.nanochat_vision_model import NanoChatVisionModel
from nanochat.vision.coco_dataset import (
    build_clip_processor,
    build_caption_tokenizer,
    create_coco_dataloader,
)
from nanochat.vision.utils import bool_arg, save_json, get_gpu_memory


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate NanoChat-V captioning quality with BLEU and CIDEr",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="Path to model.pt checkpoint (e.g. checkpoints/opt_all/model.pt)")
    p.add_argument("--config", default=None,
                   help="Path to config.json; if None, infer from checkpoint directory")
    p.add_argument("--data_root", default="data/coco",
                   help="Root directory of the COCO dataset")
    p.add_argument("--val_ann_file",
                   default="annotations/captions_val2017.json")
    p.add_argument("--val_image_dir", default="val2017")
    p.add_argument("--output_dir", default="results",
                   help="Directory to write eval_results.json")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_val_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_new_tokens", type=int, default=64,
                   help="Maximum tokens to generate per image")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Sampling temperature (1.0=multinomial, 0.0≈greedy)")
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--use_amp", type=bool_arg, default=False)
    p.add_argument("--amp_dtype", default="bf16", choices=["fp16", "bf16"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, config_path: Optional[str], device: torch.device) -> NanoChatVisionModel:
    """
    Load a NanoChatVisionModel from a checkpoint file.

    The config.json in the same directory as the checkpoint is used to reconstruct
    the exact model architecture that was trained.  This ensures the checkpoint's
    state_dict matches the model structure.
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(checkpoint_path), "config.json")

    if os.path.exists(config_path):
        cfg = VisionExperimentConfig.load(config_path)
        model_cfg = cfg.model
    else:
        print(f"WARNING: config.json not found at {config_path}, using defaults")
        model_cfg = VisionModelConfig()

    model = NanoChatVisionModel(model_cfg).to(device)

    # Load the saved state dictionary
    state = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    print(f"Model loaded from {checkpoint_path}")
    return model, model_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Caption generation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_captions(
    model: NanoChatVisionModel,
    val_loader,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    autocast_ctx,
) -> Dict[int, str]:
    """
    Run model.generate() on every batch in val_loader and collect results.

    Returns:
        image_id → generated_caption (dict)

    The image_id keys match the COCO annotation format so we can look up
    reference captions from the annotation JSON for metric computation.
    """
    generated: Dict[int, str] = {}

    print("Generating captions …")
    num_batches = len(val_loader)

    for batch_idx, batch in enumerate(val_loader):
        if batch_idx % 20 == 0:
            print(f"  batch {batch_idx}/{num_batches} …")

        pixel_values = batch["pixel_values"].to(device)
        image_ids    = batch["image_id"]  # list of ints

        with autocast_ctx():
            # generate() returns (B, max_new_tokens) new token ids
            new_token_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
                temperature=max(temperature, 1e-6),  # avoid 0.0 → multinomial crash
                top_k=top_k,
                use_visual_kv_cache=True,
            )

        # Decode each generated sequence to text
        for i, img_id in enumerate(image_ids):
            ids_i = new_token_ids[i].tolist()

            # Stop at EOS (id=50256 for GPT-2 BPE); strip everything after it
            try:
                eos_pos = ids_i.index(tokenizer.eos_token_id)
                ids_i   = ids_i[:eos_pos]
            except ValueError:
                pass   # no EOS found → use all tokens

            caption = tokenizer.decode(ids_i, skip_special_tokens=True).strip()
            generated[int(img_id)] = caption

    print(f"  Generated {len(generated)} captions")
    return generated


# ─────────────────────────────────────────────────────────────────────────────
# BLEU computation (NLTK — no pycocoevalcap required)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bleu(
    generated: Dict[int, str],
    references: Dict[int, List[str]],
) -> Dict[str, float]:
    """
    Compute corpus-level BLEU-1 through BLEU-4 using NLTK.

    BLEU is a precision-based metric: it measures what fraction of n-grams in the
    hypothesis appear in any reference.  The brevity penalty discourages very short
    hypotheses.

    We use corpus_bleu (not sentence_bleu) to compute a single score over the whole
    val set, which is more stable than averaging per-sentence scores.

    Returns:
        {"bleu1": float, "bleu2": float, "bleu3": float, "bleu4": float}
    """
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    except ImportError:
        print("BLEU: nltk not installed → skipping (pip install nltk)")
        return {"bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0}

    # Build parallel lists expected by corpus_bleu
    all_refs  = []   # list of list of list of tokens (multiple references per image)
    all_hyps  = []   # list of list of tokens (one hypothesis per image)
    smooth    = SmoothingFunction().method1

    for img_id, hyp in generated.items():
        refs = references.get(img_id, [""])
        # Tokenise by whitespace (simple but consistent with standard COCO eval)
        hyp_tokens  = hyp.lower().split()
        ref_tokens  = [r.lower().split() for r in refs]
        all_refs.append(ref_tokens)
        all_hyps.append(hyp_tokens)

    # corpus_bleu weights: (1,0,0,0)=BLEU-1, (0.5,0.5,0,0)=BLEU-2, etc.
    bleu1 = corpus_bleu(all_refs, all_hyps, weights=(1, 0, 0, 0),       smoothing_function=smooth)
    bleu2 = corpus_bleu(all_refs, all_hyps, weights=(0.5, 0.5, 0, 0),   smoothing_function=smooth)
    bleu3 = corpus_bleu(all_refs, all_hyps, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smooth)
    bleu4 = corpus_bleu(all_refs, all_hyps, weights=(0.25,)*4,           smoothing_function=smooth)

    return {
        "bleu1": round(bleu1 * 100, 2),
        "bleu2": round(bleu2 * 100, 2),
        "bleu3": round(bleu3 * 100, 2),
        "bleu4": round(bleu4 * 100, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CIDEr computation (pycocoevalcap)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cider(
    generated: Dict[int, str],
    ann_file: str,
) -> float:
    """
    Compute CIDEr using pycocoevalcap (the official COCO evaluation toolkit).

    CIDEr consensus-scores captions using TF-IDF weighted n-gram matching against
    ALL reference captions for each image.  Captions that contain rare but informative
    n-grams (specific objects, attributes) score higher than generic descriptions.

    Requires:
        pip install pycocotools pycocoevalcap

    Args:
        generated: {image_id: caption_string}
        ann_file:  path to captions_val2017.json

    Returns:
        CIDEr score (typically 0–120 for COCO; higher is better)
    """
    try:
        from pycocotools.coco import COCO
        from pycocoevalcap.eval import COCOEvalCap
    except ImportError:
        print("CIDEr: pycocoevalcap not installed → skipping "
              "(pip install pycocotools pycocoevalcap)")
        return 0.0

    # COCO evaluation toolkit expects results in the official submission format:
    # [{"image_id": int, "caption": str}, ...]
    results = [{"image_id": img_id, "caption": cap}
               for img_id, cap in generated.items()]

    coco_gt   = COCO(ann_file)
    coco_res  = coco_gt.loadRes(results)
    evaluator = COCOEvalCap(coco_gt, coco_res)
    evaluator.params["image_id"] = list(generated.keys())
    evaluator.evaluate()

    # evaluator.eval is a dict: {"Bleu_1": ..., "CIDEr": ..., ...}
    return round(evaluator.eval.get("CIDEr", 0.0), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Reference caption loader
# ─────────────────────────────────────────────────────────────────────────────

def load_references(ann_file: str, image_ids: Optional[List[int]] = None) -> Dict[int, List[str]]:
    """
    Load reference captions from a COCO annotations JSON file.

    Returns:
        {image_id: [caption1, caption2, ...]}

    Typically each image has 5 reference captions in COCO 2017.
    """
    with open(ann_file, "r") as f:
        data = json.load(f)

    refs: Dict[int, List[str]] = {}
    for ann in data["annotations"]:
        img_id  = ann["image_id"]
        if image_ids is not None and img_id not in image_ids:
            continue
        if img_id not in refs:
            refs[img_id] = []
        refs[img_id].append(ann["caption"])

    return refs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import contextlib
    args   = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    model, model_cfg = load_model(args.checkpoint, args.config, device)

    # ── Tokenizer + processor ─────────────────────────────────────────────────
    processor = build_clip_processor()
    tokenizer = build_caption_tokenizer()

    # ── DataLoader ────────────────────────────────────────────────────────────
    ann_file  = os.path.join(args.data_root, args.val_ann_file)
    image_dir = os.path.join(args.data_root, args.val_image_dir)

    val_loader = create_coco_dataloader(
        ann_file=ann_file,
        image_dir=image_dir,
        processor=processor,
        tokenizer=tokenizer,
        max_caption_len=model_cfg.max_caption_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
        max_samples=args.max_val_samples,
        split="val",
    )

    # ── AMP autocast ─────────────────────────────────────────────────────────
    if args.use_amp and device.type == "cuda":
        amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
        autocast_ctx = lambda: torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        autocast_ctx = contextlib.nullcontext

    # ── Generate ──────────────────────────────────────────────────────────────
    generated = generate_captions(
        model=model,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        autocast_ctx=autocast_ctx,
    )

    # ── Load reference captions ────────────────────────────────────────────────
    print("Loading reference captions …")
    image_ids  = list(generated.keys())
    references = load_references(ann_file, image_ids=image_ids)

    # ── BLEU ──────────────────────────────────────────────────────────────────
    print("Computing BLEU …")
    bleu_scores = compute_bleu(generated, references)
    print(f"  BLEU-1={bleu_scores['bleu1']:.2f}  "
          f"BLEU-2={bleu_scores['bleu2']:.2f}  "
          f"BLEU-3={bleu_scores['bleu3']:.2f}  "
          f"BLEU-4={bleu_scores['bleu4']:.2f}")

    # ── CIDEr ─────────────────────────────────────────────────────────────────
    print("Computing CIDEr …")
    try:
        cider_score = compute_cider(generated, ann_file)
        print(f"  CIDEr={cider_score:.3f}")
    except Exception as e:
        print(f"  CIDEr computation failed: {e}")
        cider_score = 0.0

    # ── GPU memory ────────────────────────────────────────────────────────────
    gpu_mem = get_gpu_memory() if device.type == "cuda" else {}

    # ── Save results ──────────────────────────────────────────────────────────
    results = {
        "checkpoint":      args.checkpoint,
        "num_images":      len(generated),
        "max_new_tokens":  args.max_new_tokens,
        "temperature":     args.temperature,
        "top_k":           args.top_k,
        **bleu_scores,
        "cider":           cider_score,
        "peak_mem_mb":     gpu_mem.get("peak_allocated_mb", 0.0),
        # Store a sample of generated captions for qualitative inspection
        "samples": [
            {"image_id": img_id, "generated": cap, "references": references.get(img_id, [])}
            for img_id, cap in list(generated.items())[:20]
        ],
    }

    out_path = os.path.join(args.output_dir, "eval_results.json")
    save_json(results, out_path)
    print(f"\nResults saved → {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print(f"  Images evaluated: {len(generated)}")
    print(f"  BLEU-1:  {bleu_scores['bleu1']:.2f}")
    print(f"  BLEU-2:  {bleu_scores['bleu2']:.2f}")
    print(f"  BLEU-3:  {bleu_scores['bleu3']:.2f}")
    print(f"  BLEU-4:  {bleu_scores['bleu4']:.2f}")
    print(f"  CIDEr:   {cider_score:.3f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
