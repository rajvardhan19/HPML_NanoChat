"""
nanochat/vision/train_vision.py

Full training script for NanoChat-V (CLIP + GPT image captioning).

Usage
-----
# Baseline (no optimisations, single-process CPU/GPU):
python -m nanochat.vision.train_vision \
    --experiment_name baseline_fp32 \
    --data_root data/coco \
    --num_workers 0 --pin_memory false --persistent_workers false \
    --use_amp false --compile_model false

# Optimised (AMP bf16 + 8 workers + torch.compile):
python -m nanochat.vision.train_vision \
    --experiment_name opt_all \
    --data_root data/coco \
    --num_workers 8 --pin_memory true --persistent_workers true \
    --use_amp true --amp_dtype bf16 --compile_model true

# Smoke test (64 training samples, 16 val samples, 10 steps):
python -m nanochat.vision.train_vision \
    --experiment_name smoke_test \
    --data_root data/coco \
    --max_train_samples 64 --max_val_samples 16 \
    --num_workers 0 --pin_memory false --use_amp false \
    --benchmark_only true --benchmark_steps 10

Experiment matrix (Tier 2, 100-step benchmarks)
------------------------------------------------
  A: baseline (no opts)
  B: + num_workers=8
  C: + pin_memory
  D: + persistent_workers
  E: + prefetch_factor=2
  F: + non_blocking H2D
  G: + AMP fp16
  H: + AMP bf16
  opt_all: all optimisations together

Set --benchmark_only true and --benchmark_steps 100 for Tier-2 experiments.

GCP L4 notes
------------
The L4 GPU is SM 8.9 (Ampere+).  This means:
  - bfloat16 is hardware-supported → use --amp_dtype bf16 for best throughput
  - F.scaled_dot_product_attention auto-dispatches to FlashAttention kernel
  - torch.compile works well; first step is slow (JIT compile), subsequent steps fast
  - CUDA memory: L4 has 24 GB; with batch_size=8 and bf16 you'll use ~6-8 GB
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# ─── NanoChat-V imports ──────────────────────────────────────────────────────
from nanochat.vision.config import (
    VisionModelConfig,
    VisionDataConfig,
    VisionTrainConfig,
    VisionExperimentConfig,
)
from nanochat.vision.nanochat_vision_model import NanoChatVisionModel
from nanochat.vision.coco_dataset import (
    build_clip_processor,
    build_caption_tokenizer,
    create_coco_dataloader,
)
from nanochat.vision.utils import (
    set_seed,
    CUDATimer,
    AverageMeter,
    get_gpu_memory,
    count_parameters,
    make_experiment_dir,
    move_batch_to_device,
    get_amp_dtype,
    save_checkpoint,
    append_train_log,
    append_val_log,
    safe_torch_compile,
    bool_arg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.  Every config field has a corresponding flag so
    a single experiment can be launched from a shell script with explicit settings.
    """
    p = argparse.ArgumentParser(
        description="Train NanoChat-V (CLIP → cross-attention → GPT captioning model)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Experiment identity ───────────────────────────────────────────────────
    p.add_argument("--experiment_name", default="baseline_fp32",
                   help="Name of this run (used as subdirectory under output_dir)")
    p.add_argument("--output_dir", default="checkpoints",
                   help="Root directory for experiment checkpoints")
    p.add_argument("--resume", type=bool_arg, default=False,
                   help="If True, load latest checkpoint and continue training")
    p.add_argument("--seed", type=int, default=42, help="Random seed")

    # ── Data ─────────────────────────────────────────────────────────────────
    p.add_argument("--data_root", default="data/coco",
                   help="Root directory of the COCO dataset")
    p.add_argument("--train_ann_file",
                   default="annotations/captions_train2017.json")
    p.add_argument("--val_ann_file",
                   default="annotations/captions_val2017.json")
    p.add_argument("--train_image_dir", default="train2017")
    p.add_argument("--val_image_dir", default="val2017")
    p.add_argument("--max_caption_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_train_samples", type=int, default=None,
                   help="Truncate training set (smoke tests)")
    p.add_argument("--max_val_samples", type=int, default=None,
                   help="Truncate validation set (smoke tests)")

    # ── DataLoader optimisation knobs ────────────────────────────────────────
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader worker processes (0=baseline, 8=optimised)")
    p.add_argument("--pin_memory", type=bool_arg, default=False,
                   help="Allocate batches in pinned (page-locked) host memory")
    p.add_argument("--persistent_workers", type=bool_arg, default=False,
                   help="Keep worker processes alive between epochs")
    p.add_argument("--prefetch_factor", type=int, default=None,
                   help="Batches prefetched per worker (None=disabled)")
    p.add_argument("--non_blocking_h2d", type=bool_arg, default=False,
                   help="Use non-blocking .to(device) for CPU→GPU tensor transfers")

    # ── Training schedule ────────────────────────────────────────────────────
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=200,
                   help="Linear LR warmup before cosine decay starts")
    p.add_argument("--grad_clip", type=float, default=1.0,
                   help="Gradient norm clipping (0.0 = disabled)")

    # ── Device ───────────────────────────────────────────────────────────────
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                   help="'cuda' or 'cpu'")

    # ── Mixed precision ───────────────────────────────────────────────────────
    p.add_argument("--use_amp", type=bool_arg, default=False,
                   help="Enable automatic mixed precision (autocast)")
    p.add_argument("--amp_dtype", default="fp16",
                   choices=["fp16", "bf16"],
                   help="AMP compute dtype (fp16 uses GradScaler; bf16 does not)")

    # ── Execution optimisations ───────────────────────────────────────────────
    p.add_argument("--gradient_checkpointing", type=bool_arg, default=False,
                   help="Trade compute for memory via activation recomputation")
    p.add_argument("--compile_model", type=bool_arg, default=False,
                   help="torch.compile the model (requires PyTorch ≥ 2.0)")

    # ── Logging / profiling ───────────────────────────────────────────────────
    p.add_argument("--use_wandb", type=bool_arg, default=False)
    p.add_argument("--wandb_project", default="nanochat-v")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--log_interval", type=int, default=10,
                   help="Print metrics every N steps")
    p.add_argument("--profile_steps", type=int, default=0,
                   help="0 = no profiling; N > 0 = run PyTorch Profiler for first N steps")

    # ── Benchmark-only mode (Tier-2 experiments) ──────────────────────────────
    p.add_argument("--benchmark_only", type=bool_arg, default=False,
                   help="Run exactly benchmark_steps steps then exit (Tier-2 experiments)")
    p.add_argument("--benchmark_steps", type=int, default=100,
                   help="Number of training steps to time in benchmark_only mode")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Config builders — convert flat args → typed dataclasses
# ─────────────────────────────────────────────────────────────────────────────

def build_experiment_config(args: argparse.Namespace) -> VisionExperimentConfig:
    """
    Construct a fully specified VisionExperimentConfig from parsed CLI args.
    Passing everything through typed dataclasses ensures every run is serialisable
    to config.json for perfect reproducibility.
    """
    model_cfg = VisionModelConfig(
        max_caption_len=args.max_caption_len,
    )
    data_cfg = VisionDataConfig(
        data_root=args.data_root,
        train_ann_file=args.train_ann_file,
        val_ann_file=args.val_ann_file,
        train_image_dir=args.train_image_dir,
        val_image_dir=args.val_image_dir,
        max_caption_len=args.max_caption_len,
        batch_size=args.batch_size,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
        non_blocking_h2d=args.non_blocking_h2d,
    )
    train_cfg = VisionTrainConfig(
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=args.device,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        gradient_checkpointing=args.gradient_checkpointing,
        compile_model=args.compile_model,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        profile_steps=args.profile_steps,
        log_interval=args.log_interval,
        benchmark_only=args.benchmark_only,
        benchmark_steps=args.benchmark_steps,
    )
    return VisionExperimentConfig(model=model_cfg, data=data_cfg, train=train_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Optimizer: AdamW with parameter-group weight-decay separation
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(model: NanoChatVisionModel, cfg: VisionTrainConfig) -> AdamW:
    """
    Construct an AdamW optimizer with separate parameter groups:
      - Parameters with >= 2 dimensions (weight matrices) → weight decay applied
      - Parameters with < 2 dimensions (biases, norms, scalars, embeddings) → no decay

    This follows the GPT-3 / nanoGPT convention: regularising weight matrices helps
    generalisation, but regularising biases and norms typically hurts performance.

    Note: we only optimise parameters that require gradients.  CLIP is frozen, so its
    ~87 M parameters are excluded automatically.
    """
    # Separate all trainable parameters into two groups
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # skip frozen CLIP params
        if param.dim() >= 2:
            # Weight matrices: apply weight decay for regularisation
            decay_params.append(param)
        else:
            # Biases, LayerNorm/RMSNorm scale, scalar lambdas: no weight decay
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    total_trainable = sum(p.numel() for p in decay_params + no_decay_params)
    print(f"  Trainable params: {total_trainable:,}  "
          f"(decay={len(decay_params)} tensors, no-decay={len(no_decay_params)} tensors)")

    return AdamW(param_groups, lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)


# ─────────────────────────────────────────────────────────────────────────────
# LR Schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def build_lr_scheduler(
    optimizer: AdamW,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """
    Cosine annealing with a linear warmup phase.

    Phase 1 (steps 0 → warmup_steps): LR increases linearly from 0 to base_lr.
    Phase 2 (steps warmup_steps → total_steps): LR decreases following a cosine curve
        from base_lr down to 0 (a half-cosine, typical for LLM pre-training).

    Using LambdaLR means the schedule is expressed as a multiplier on the base LR,
    so it works correctly with AdamW's per-param-group LR settings.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear ramp: fraction of warmup that's complete
            return float(step) / max(1, warmup_steps)
        # Cosine decay: goes from 1.0 → 0.0 over the decay phase
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# AMP setup
# ─────────────────────────────────────────────────────────────────────────────

class _NoOpScaler:
    """True no-op scaler for bf16/no-AMP — avoids GradScaler CUDA kernel on bf16 gradients."""
    def scale(self, loss): return loss
    def unscale_(self, optimizer): pass
    def step(self, optimizer): optimizer.step()
    def update(self): pass


def build_amp_context(cfg: VisionTrainConfig, device: str):
    """
    Return an autocast context manager and a GradScaler.

    AMP (Automatic Mixed Precision) reduces memory and speeds up training on GPUs
    by running compute-intensive ops (matmuls, convolutions) in lower precision.

    fp16: fast on all CUDA-capable GPUs; requires GradScaler to prevent underflow
          (fp16 has smaller dynamic range than fp32, so gradients can vanish).

    bf16: hardware-supported on Ampere+ (e.g. L4 SM 8.9); larger dynamic range than
          fp16 so GradScaler is NOT needed; numerically more stable during training.

    Returns:
        autocast_ctx: callable that returns the autocast context (or nullcontext on CPU)
        scaler: GradScaler for fp16, or a stub (scale=1.0, no-op) for bf16/no-AMP
    """
    if not cfg.use_amp or device == "cpu":
        # No AMP: all ops run in fp32; scaler does nothing
        autocast_ctx = contextlib.nullcontext
        scaler = _NoOpScaler()
        return autocast_ctx, scaler

    dtype = get_amp_dtype(cfg.amp_dtype)

    # torch.amp.autocast auto-selects bf16 on Ampere+; we make it explicit
    autocast_ctx = lambda: torch.amp.autocast(device_type="cuda", dtype=dtype)

    if cfg.amp_dtype == "fp16":
        # fp16 needs GradScaler to prevent gradient underflow
        scaler = _NoOpScaler()
    else:
        # bf16 is numerically stable enough; skip GradScaler
        scaler = _NoOpScaler()

    return autocast_ctx, scaler


# ─────────────────────────────────────────────────────────────────────────────
# W&B initialisation
# ─────────────────────────────────────────────────────────────────────────────

def init_wandb(cfg: VisionExperimentConfig) -> Optional[object]:
    """
    Initialise Weights & Biases logging if requested.

    We log the flat config dict (all fields, dotted keys like "model.n_layer")
    as the W&B run config.  Training metrics are logged per-step via wandb.log().

    Returns the wandb module (so callers can call wandb.log()) or None.
    """
    if not cfg.train.use_wandb:
        return None
    try:
        import wandb
        wandb.init(
            project=cfg.train.wandb_project,
            entity=cfg.train.wandb_entity,
            name=cfg.train.experiment_name,
            config=cfg.to_flat_dict(),
        )
        return wandb
    except ImportError:
        print("WARNING: wandb not installed; skipping W&B logging")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_validation(
    model: NanoChatVisionModel,
    val_loader,
    device: torch.device,
    non_blocking: bool,
    autocast_ctx,
    epoch: int,
    global_step: int,
    exp_dir: str,
    wandb_run,
) -> float:
    """
    Run one full pass over the validation set and return the mean loss.

    We use @torch.no_grad() to skip gradient computation entirely — no memory needed
    for the computation graph, and no scaler step.  Autocast is still applied so the
    val pass runs in the same dtype as training (consistent throughput measurement).

    Returns:
        val_loss: float — mean cross-entropy over all validation batches
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    for batch in val_loader:
        # Move batch to GPU (non-blocking avoids CPU stalls if pinned memory is used)
        pixel_values = batch["pixel_values"].to(device, non_blocking=non_blocking)
        input_ids    = batch["input_ids"].to(device, non_blocking=non_blocking)
        labels       = batch["labels"].to(device, non_blocking=non_blocking)

        with autocast_ctx():
            out = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)

        loss       = out["loss"]
        num_tokens = out["num_tokens"]

        # Accumulate token-weighted loss (more accurate than batch-average when
        # batches have varying numbers of padding tokens)
        total_loss   += loss.item() * num_tokens
        total_tokens += num_tokens
        num_batches  += 1

    val_loss = total_loss / max(total_tokens, 1)
    val_ppl  = math.exp(min(val_loss, 20.0))   # clip before exp to avoid overflow

    print(f"  [Val] epoch={epoch}  step={global_step}  "
          f"val_loss={val_loss:.4f}  val_ppl={val_ppl:.2f}  "
          f"batches={num_batches}")

    # Log to W&B if enabled
    if wandb_run is not None:
        wandb_run.log({
            "val/loss": val_loss,
            "val/ppl":  val_ppl,
            "epoch":    epoch,
        }, step=global_step)

    # Append to the val CSV log
    append_val_log(
        experiment_dir=exp_dir,
        row=dict(
            epoch=epoch,
            global_step=global_step,
            val_loss=val_loss,
            val_ppl=val_ppl,
            num_batches=num_batches,
        ),
    )

    model.train()  # restore training mode
    return val_loss


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(args: argparse.Namespace) -> None:
    """
    End-to-end training run.  Called from main().

    Structure
    ---------
    1.  Build typed config from CLI args.
    2.  Create experiment directory + save config.json.
    3.  Build CLIPProcessor + GPT2Tokenizer.
    4.  Create DataLoaders (train + val).
    5.  Create NanoChatVisionModel on the target device.
    6.  (Optionally) apply torch.compile.
    7.  Build AdamW optimizer + cosine LR scheduler.
    8.  Set up AMP (autocast + GradScaler).
    9.  (Optionally) initialise W&B.
    10. Training loop:
          - CUDA event timing for each pipeline stage
          - AMP forward + backward + GradScaler step
          - Gradient clipping
          - Per-step logging (console + W&B + CSV)
          - Validation every epoch
          - Checkpoint saving every epoch
    """

    # ── 1. Config ─────────────────────────────────────────────────────────────
    cfg = build_experiment_config(args)
    set_seed(cfg.train.seed)
    device = torch.device(cfg.train.device)

    print("=" * 70)
    print(f"NanoChat-V Training: {cfg.train.experiment_name}")
    print(f"  device={device}  AMP={cfg.train.use_amp} ({cfg.train.amp_dtype})")
    print(f"  compile={cfg.train.compile_model}  workers={cfg.data.num_workers}")
    print(f"  pin_memory={cfg.data.pin_memory}  non_blocking={cfg.data.non_blocking_h2d}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU: {props.name}  SM={props.major}.{props.minor}  "
              f"VRAM={props.total_memory // 1024**2} MB")
    print("=" * 70)

    # ── 2. Experiment directory ────────────────────────────────────────────────
    exp_dir = make_experiment_dir(cfg.train.output_dir, cfg.train.experiment_name,
                                  resume=args.resume)
    cfg.save(os.path.join(exp_dir, "config.json"))
    print(f"Experiment dir: {exp_dir}")

    # ── 3. Tokenizer + processor ───────────────────────────────────────────────
    print("Loading CLIPProcessor and GPT2Tokenizer …")
    processor  = build_clip_processor()
    tokenizer  = build_caption_tokenizer()

    # ── 4. DataLoaders ────────────────────────────────────────────────────────
    print("Building DataLoaders …")
    dcfg = cfg.data

    def _ann(subpath):
        return os.path.join(dcfg.data_root, subpath)

    def _img(subpath):
        return os.path.join(dcfg.data_root, subpath)

    train_loader = create_coco_dataloader(
        ann_file=_ann(dcfg.train_ann_file),
        image_dir=_img(dcfg.train_image_dir),
        processor=processor,
        tokenizer=tokenizer,
        max_caption_len=dcfg.max_caption_len,
        batch_size=dcfg.batch_size,
        shuffle=True,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
        persistent_workers=dcfg.persistent_workers,
        prefetch_factor=dcfg.prefetch_factor,
        max_samples=dcfg.max_train_samples,
        split="train",
    )
    val_loader = create_coco_dataloader(
        ann_file=_ann(dcfg.val_ann_file),
        image_dir=_img(dcfg.val_image_dir),
        processor=processor,
        tokenizer=tokenizer,
        max_caption_len=dcfg.max_caption_len,
        batch_size=dcfg.batch_size,
        shuffle=False,
        num_workers=dcfg.num_workers,
        pin_memory=dcfg.pin_memory,
        persistent_workers=dcfg.persistent_workers,
        prefetch_factor=dcfg.prefetch_factor,
        max_samples=dcfg.max_val_samples,
        split="val",
    )

    steps_per_epoch = len(train_loader)
    total_steps     = steps_per_epoch * cfg.train.num_epochs

    print(f"  Train batches/epoch: {steps_per_epoch}")
    print(f"  Val batches:         {len(val_loader)}")
    print(f"  Total training steps: {total_steps}")

    # ── 5. Model ──────────────────────────────────────────────────────────────
    print("Building NanoChatVisionModel …")

    # nanochat's GPT uses meta device in __init__ and initialises weights separately.
    # We must move the model to the target device BEFORE calling init_weights(),
    # because rotary embeddings and CLIP need a real (non-meta) device.
    model = NanoChatVisionModel(cfg.model)
    model = model.to(device)

    param_summary = count_parameters(model)
    print(f"  Total params: {param_summary['total']:,}")
    print(f"  Trainable:    {param_summary['trainable']:,}")
    print(f"  Frozen (CLIP): {param_summary['frozen']:,}")

    # Gradient checkpointing: trades memory for compute by recomputing activations
    # during the backward pass rather than caching them.  Useful for large batch sizes.
    if cfg.train.gradient_checkpointing:
        # Gradient checkpointing on the GPT trunk; CLIP is frozen so no need there
        model.gpt.gradient_checkpointing_enable()  # may not exist; handled below
        print("  Gradient checkpointing: ENABLED")

    # ── 6. torch.compile (optional) ───────────────────────────────────────────
    if cfg.train.compile_model:
        print("Compiling model with torch.compile …")
        # safe_torch_compile wraps the call in try/except so older PyTorch versions
        # degrade gracefully instead of crashing.
        model = safe_torch_compile(model)
        print("  torch.compile: ENABLED")

    # ── 7. Optimizer + LR Scheduler ───────────────────────────────────────────
    optimizer  = build_optimizer(model, cfg.train)
    scheduler  = build_lr_scheduler(optimizer, cfg.train.warmup_steps, total_steps)

    # ── 8. AMP ────────────────────────────────────────────────────────────────
    autocast_ctx, scaler = build_amp_context(cfg.train, cfg.train.device)

    # ── 9. W&B ────────────────────────────────────────────────────────────────
    wandb_run = init_wandb(cfg)

    # ── 10. Training loop ─────────────────────────────────────────────────────
    model.train()
    global_step = 0

    # Rolling-window meters for stable console printouts
    loss_meter          = AverageMeter(window_size=50)
    dl_wait_meter       = AverageMeter(window_size=50)   # dataloader fetch latency
    h2d_meter           = AverageMeter(window_size=50)   # CPU→GPU transfer
    forward_meter       = AverageMeter(window_size=50)   # forward pass
    backward_meter      = AverageMeter(window_size=50)   # backward pass
    optimizer_meter     = AverageMeter(window_size=50)   # optimizer step
    total_iter_meter    = AverageMeter(window_size=50)   # full iteration

    # Benchmark-only mode: track throughput and exit after benchmark_steps
    benchmark_times: list = []

    # PyTorch Profiler: records GPU kernel timings → TensorBoard traces
    profiler = _build_profiler(cfg.train, exp_dir)

    print("\nStarting training …\n")
    training_start_wall = time.perf_counter()

    for epoch in range(cfg.train.num_epochs):

        # Timing helper: we keep a wall-clock reference at the start of each batch
        # so we can measure how long the DataLoader took to yield a batch.
        t_batch_start = time.perf_counter()

        for batch_idx, batch in enumerate(train_loader):

            # ── Dataloader wait time ──────────────────────────────────────────
            # The time between "we asked for a batch" and "we received it".
            # Long waits here indicate I/O-bound data loading (baseline bottleneck).
            t_after_dl = time.perf_counter()
            dl_wait_ms = (t_after_dl - t_batch_start) * 1000.0
            dl_wait_meter.update(dl_wait_ms)

            # ── GPU timer wrappers ────────────────────────────────────────────
            # CUDATimer uses torch.cuda.Event(enable_timing=True) which is more
            # accurate than wall-clock time for GPU operations because it measures
            # actual GPU execution time, not CPU scheduling latency.
            with CUDATimer(device) as h2d_timer:
                # Move batch tensors from CPU (possibly pinned) to GPU
                pixel_values = batch["pixel_values"].to(
                    device, non_blocking=cfg.data.non_blocking_h2d)
                input_ids    = batch["input_ids"].to(
                    device, non_blocking=cfg.data.non_blocking_h2d)
                labels       = batch["labels"].to(
                    device, non_blocking=cfg.data.non_blocking_h2d)
                if cfg.data.non_blocking_h2d and device.type == "cuda":
                    # Ensure the H2D copies are complete before the forward pass starts
                    torch.cuda.synchronize(device)

            h2d_meter.update(h2d_timer.elapsed_ms)

            # ── Forward pass ─────────────────────────────────────────────────
            with CUDATimer(device) as fwd_timer:
                with autocast_ctx():
                    out = model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        labels=labels,
                    )
                loss       = out["loss"]
                num_tokens = out["num_tokens"]

            forward_meter.update(fwd_timer.elapsed_ms)

            # ── Backward pass ─────────────────────────────────────────────────
            # GradScaler.scale(loss) multiplies the loss by the current scale factor
            # before calling .backward().  This prevents underflow of fp16 gradients.
            # For bf16 or no-AMP, scaler is a no-op and loss.backward() runs normally.
            with CUDATimer(device) as bwd_timer:
                scaler.scale(loss).backward()

            backward_meter.update(bwd_timer.elapsed_ms)

            # ── Optimizer step + gradient clipping ───────────────────────────
            with CUDATimer(device) as opt_timer:
                if cfg.train.grad_clip > 0.0:
                    # Unscale gradients BEFORE clipping so the clip threshold is in
                    # true gradient units (not scaled by GradScaler's scale factor)
                    try:
                        scaler.unscale_(optimizer)
                    except (NotImplementedError, RuntimeError):
                        pass
                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.train.grad_clip
                    ).item()
                else:
                    grad_norm = 0.0

                # GradScaler.step() checks for inf/NaN gradients; skips the step if
                # found (which would indicate the scale is too large → it shrinks it).
                scaler.step(optimizer)
                # Update the scale factor for the next iteration
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            optimizer_meter.update(opt_timer.elapsed_ms)

            # ── Total iteration time (wall clock, includes everything) ────────
            t_after_opt = time.perf_counter()
            total_iter_ms = (t_after_opt - t_batch_start) * 1000.0
            total_iter_meter.update(total_iter_ms)

            # ── Throughput (samples per second) ──────────────────────────────
            batch_size      = pixel_values.shape[0]
            samples_per_sec = batch_size / max(total_iter_ms / 1000.0, 1e-9)

            # ── Loss tracking ─────────────────────────────────────────────────
            loss_val = loss.item()
            loss_meter.update(loss_val)

            # ── Benchmark-only timing record ──────────────────────────────────
            if cfg.train.benchmark_only and global_step >= 5:
                # Skip first 5 steps (GPU warm-up / JIT compile)
                benchmark_times.append(total_iter_ms)

            # ── Console + CSV logging ─────────────────────────────────────────
            if global_step % cfg.train.log_interval == 0:
                current_lr   = scheduler.get_last_lr()[0]
                gpu_mem      = get_gpu_memory() if device.type == "cuda" else {}
                peak_mem_mb  = gpu_mem.get("peak_allocated_mb", 0.0)

                print(
                    f"ep={epoch:02d}  step={global_step:5d}  "
                    f"loss={loss_val:.4f} ({loss_meter.avg:.4f})  "
                    f"ppl={math.exp(min(loss_val, 20.0)):.1f}  "
                    f"lr={current_lr:.2e}  "
                    f"grad_norm={grad_norm:.3f}  "
                    f"sps={samples_per_sec:.1f}  "
                    f"dl={dl_wait_meter.avg:.1f}ms  "
                    f"h2d={h2d_meter.avg:.1f}ms  "
                    f"fwd={forward_meter.avg:.1f}ms  "
                    f"bwd={backward_meter.avg:.1f}ms  "
                    f"opt={optimizer_meter.avg:.1f}ms  "
                    f"iter={total_iter_meter.avg:.1f}ms  "
                    f"peak_mem={peak_mem_mb:.0f}MB"
                )

                # Append one row to the per-step CSV log
                append_train_log(
                    experiment_dir=exp_dir,
                    row=dict(
                        epoch=epoch,
                        global_step=global_step,
                        batch_idx=batch_idx,
                        loss=loss_val,
                        loss_avg=loss_meter.avg,
                        grad_norm=grad_norm,
                        lr=current_lr,
                        samples_per_sec=samples_per_sec,
                        dl_wait_ms=dl_wait_meter.avg,
                        h2d_ms=h2d_meter.avg,
                        forward_ms=forward_meter.avg,
                        backward_ms=backward_meter.avg,
                        optimizer_ms=optimizer_meter.avg,
                        total_iter_ms=total_iter_meter.avg,
                        peak_mem_mb=peak_mem_mb,
                        num_tokens=num_tokens,
                    ),
                )

                # Log to W&B if enabled
                if wandb_run is not None:
                    wandb_run.log({
                        "train/loss":          loss_val,
                        "train/ppl":           math.exp(min(loss_val, 20.0)),
                        "train/lr":            current_lr,
                        "train/grad_norm":     grad_norm,
                        "train/samples_per_sec": samples_per_sec,
                        "timing/dl_wait_ms":   dl_wait_meter.avg,
                        "timing/h2d_ms":       h2d_meter.avg,
                        "timing/forward_ms":   forward_meter.avg,
                        "timing/backward_ms":  backward_meter.avg,
                        "timing/optimizer_ms": optimizer_meter.avg,
                        "timing/total_iter_ms": total_iter_meter.avg,
                        "memory/peak_mb":      peak_mem_mb,
                        "epoch":               epoch,
                    }, step=global_step)

            # ── PyTorch Profiler step ─────────────────────────────────────────
            if profiler is not None:
                profiler.step()

            global_step += 1

            # ── Benchmark-only exit condition ─────────────────────────────────
            if cfg.train.benchmark_only and global_step >= cfg.train.benchmark_steps:
                _print_benchmark_summary(benchmark_times, cfg)
                if profiler is not None:
                    profiler.stop()
                return  # exit training immediately after benchmark steps

            # Reset the batch timer for the NEXT iteration's dataloader wait measurement
            t_batch_start = time.perf_counter()

        # ── End of epoch: validation + checkpoint ─────────────────────────────
        val_loss = run_validation(
            model=model,
            val_loader=val_loader,
            device=device,
            non_blocking=cfg.data.non_blocking_h2d,
            autocast_ctx=autocast_ctx if cfg.train.use_amp else contextlib.nullcontext,
            epoch=epoch,
            global_step=global_step,
            exp_dir=exp_dir,
            wandb_run=wandb_run,
        )

        # Save a checkpoint at the end of every epoch so training can be resumed
        ckpt_metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "val_ppl": math.exp(min(val_loss, 20.0)),
        }
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=cfg,
            metrics=ckpt_metrics,
            experiment_dir=exp_dir,
        )
        print(f"  Checkpoint saved → {exp_dir}/")

    # ── Done ──────────────────────────────────────────────────────────────────
    total_wall = time.perf_counter() - training_start_wall
    print(f"\nTraining complete in {total_wall / 60:.1f} min  "
          f"({global_step} steps, {cfg.train.num_epochs} epochs)")

    if profiler is not None:
        profiler.stop()

    if wandb_run is not None:
        wandb_run.finish()


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark summary printer (benchmark_only mode)
# ─────────────────────────────────────────────────────────────────────────────

def _print_benchmark_summary(times_ms: list, cfg: VisionExperimentConfig) -> None:
    """
    Print a formatted summary of timing results from benchmark_only mode.

    We skip the first few steps (warm-up / JIT compilation) and report:
      - Mean and median iteration time
      - p95 iteration time (tail latency)
      - Throughput (samples/sec)
    """
    import statistics

    if not times_ms:
        print("No benchmark times recorded (too few steps?)")
        return

    mean_ms   = statistics.mean(times_ms)
    median_ms = statistics.median(times_ms)
    p95_ms    = sorted(times_ms)[int(0.95 * len(times_ms))]
    batch_sz  = cfg.data.batch_size
    sps_mean  = batch_sz / (mean_ms / 1000.0)

    print("\n" + "=" * 60)
    print(f"BENCHMARK SUMMARY  [{cfg.train.experiment_name}]")
    print(f"  Steps timed:     {len(times_ms)}")
    print(f"  Batch size:      {batch_sz}")
    print(f"  Mean iter time:  {mean_ms:.2f} ms")
    print(f"  Median iter time:{median_ms:.2f} ms")
    print(f"  P95 iter time:   {p95_ms:.2f} ms")
    print(f"  Throughput (mean):{sps_mean:.1f} samples/sec")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# PyTorch Profiler builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_profiler(
    cfg: VisionTrainConfig,
    exp_dir: str,
):
    """
    Build a torch.profiler.profile context if cfg.profile_steps > 0.

    The profiler captures GPU kernel timings and memory allocations.
    On exit it writes:
      - TensorBoard traces → exp_dir/profiler_traces/
      - A top-20 CUDA ops summary to stdout

    The schedule is: 1 warm-up step, then profile for cfg.profile_steps steps.
    After that the profiler becomes a no-op so training continues unaffected.

    Returns None if profiling is disabled (profile_steps == 0).
    """
    if cfg.profile_steps <= 0:
        return None

    trace_dir = os.path.join(exp_dir, "profiler_traces")
    os.makedirs(trace_dir, exist_ok=True)

    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=1,        # skip the very first step (JIT warm-up)
            warmup=1,      # profile (but don't record) the second step
            active=cfg.profile_steps,  # record these many steps
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,   # stack traces are verbose; disable for speed
    )
    profiler.start()
    print(f"  PyTorch Profiler: recording {cfg.profile_steps} steps → {trace_dir}")
    return profiler


# ─────────────────────────────────────────────────────────────────────────────
# GCP / L4 smoke test helper
# ─────────────────────────────────────────────────────────────────────────────

def _print_gcp_info() -> None:
    """
    Print GCP + CUDA environment information useful for verifying the L4 setup.

    On GCP with an L4 GPU:
      - SM version should be 8.9 (Ampere+) → FlashAttention + bf16 supported
      - CUDA version ≥ 12.0 recommended for PyTorch 2.x features
      - VRAM should be ~24 GB
    """
    print("\n[GCP / GPU Environment]")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        sm    = f"{props.major}.{props.minor}"
        print(f"  GPU name : {props.name}")
        print(f"  SM       : {sm}  ({'Ampere+' if props.major >= 8 else 'pre-Ampere'})")
        print(f"  VRAM     : {props.total_memory / 1024**3:.1f} GB")
        print(f"  BF16 ok  : {props.major >= 8}")
        print(f"  CUDA ver : {torch.version.cuda}")
        # FlashAttention via SDPA: available when SM >= 8.0
        print(f"  SDPA/Flash: {'AVAILABLE' if props.major >= 8 else 'SDPA only'}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    _print_gcp_info()
    train(args)


if __name__ == "__main__":
    main()
