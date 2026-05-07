"""
nanochat/vision/benchmark_vision.py

Standalone throughput and latency benchmarking for the NanoChat-V experiment matrix.

What this script measures
-------------------------
For each of the 8 experiments (A–H) plus opt_all, we run exactly benchmark_steps
training steps and measure:
  - Training throughput   (samples/sec)
  - Iteration latency     (mean / median / p95 ms)
  - Peak GPU memory       (MB)
  - Dataloader wait time  (mean ms / p95 ms)
  - H2D copy time         (mean ms)
  - Forward pass time     (mean ms)
  - Backward pass time    (mean ms)

In addition, we run inference throughput at batch sizes [1, 4, 16]:
  - Tokens generated / sec at each batch size
  - Inference latency p50 and p95

Results are saved as:
  results/benchmark_summary.csv
  results/benchmark_summary.json
  results/inference_summary.csv

Usage
-----
# Full benchmark across all experiments:
python -m nanochat.vision.benchmark_vision \
    --data_root data/coco \
    --benchmark_steps 100 \
    --warmup_steps 10

# Quick smoke test (CPU, 4 steps, batch_size 2):
python -m nanochat.vision.benchmark_vision \
    --data_root data/coco \
    --benchmark_steps 4 \
    --warmup_steps 2 \
    --device cpu \
    --batch_size 2 \
    --max_samples 64

Experiment matrix
-----------------
Each experiment toggles exactly one optimisation relative to the baseline:

  A  – Baseline (no optimisations)
  B  – + num_workers=8
  C  – + pin_memory=True
  D  – + persistent_workers=True
  E  – + prefetch_factor=2
  F  – + non_blocking H2D transfer
  G  – + AMP fp16
  H  – + AMP bf16
  opt_all – all of B+C+D+E+F+H together
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn

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
    get_amp_dtype,
    move_batch_to_device,
    bool_arg,
    write_csv,
    save_json,
)
from nanochat.vision.train_vision import build_amp_context


# ─────────────────────────────────────────────────────────────────────────────
# Experiment specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExperimentSpec:
    """
    Specifies which optimisation knobs are active for one benchmark run.
    Each field overrides the corresponding field in VisionDataConfig / VisionTrainConfig.
    """
    name: str
    num_workers: int       = 0
    pin_memory: bool       = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    non_blocking_h2d: bool = False
    use_amp: bool          = False
    amp_dtype: str         = "fp16"


# The canonical 8-experiment + opt_all matrix from the spec
EXPERIMENT_MATRIX: List[ExperimentSpec] = [
    ExperimentSpec(name="A_baseline",              num_workers=0, pin_memory=False,
                   persistent_workers=False, prefetch_factor=None,
                   non_blocking_h2d=False, use_amp=False),
    ExperimentSpec(name="B_workers8",              num_workers=8, pin_memory=False,
                   persistent_workers=False, prefetch_factor=None,
                   non_blocking_h2d=False, use_amp=False),
    ExperimentSpec(name="C_pin_memory",            num_workers=0, pin_memory=True,
                   persistent_workers=False, prefetch_factor=None,
                   non_blocking_h2d=False, use_amp=False),
    ExperimentSpec(name="D_persistent_workers",    num_workers=0, pin_memory=False,
                   persistent_workers=True, prefetch_factor=None,
                   non_blocking_h2d=False, use_amp=False),
    ExperimentSpec(name="E_prefetch",              num_workers=4, pin_memory=False,
                   persistent_workers=False, prefetch_factor=2,
                   non_blocking_h2d=False, use_amp=False),
    ExperimentSpec(name="F_non_blocking",          num_workers=0, pin_memory=True,
                   persistent_workers=False, prefetch_factor=None,
                   non_blocking_h2d=True, use_amp=False),
    ExperimentSpec(name="G_amp_fp16",              num_workers=0, pin_memory=False,
                   persistent_workers=False, prefetch_factor=None,
                   non_blocking_h2d=False, use_amp=True, amp_dtype="fp16"),
    ExperimentSpec(name="H_amp_bf16",              num_workers=0, pin_memory=False,
                   persistent_workers=False, prefetch_factor=None,
                   non_blocking_h2d=False, use_amp=True, amp_dtype="bf16"),
    ExperimentSpec(name="opt_all",                 num_workers=8, pin_memory=True,
                   persistent_workers=True, prefetch_factor=2,
                   non_blocking_h2d=True, use_amp=True, amp_dtype="bf16"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark NanoChat-V across all 8 experiment configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", default="data/coco")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--benchmark_steps", type=int, default=100,
                   help="Timed training steps per experiment")
    p.add_argument("--warmup_steps", type=int, default=10,
                   help="Steps to skip before timing starts (GPU warm-up)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap dataset size for quick tests")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=32,
                   help="Tokens generated per inference latency run")
    p.add_argument("--inf_batch_sizes", nargs="+", type=int, default=[1, 4, 16],
                   help="Batch sizes for inference latency measurement")
    p.add_argument("--experiments", nargs="*", default=None,
                   help="Subset of experiments to run (e.g. A_baseline opt_all); "
                        "default runs all 9")
    p.add_argument("--skip_inference", type=bool_arg, default=False,
                   help="Skip inference latency benchmarks")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Training throughput benchmark for one experiment
# ─────────────────────────────────────────────────────────────────────────────

def benchmark_training(
    spec: ExperimentSpec,
    data_root: str,
    batch_size: int,
    benchmark_steps: int,
    warmup_steps: int,
    device: torch.device,
    seed: int,
    max_samples: Optional[int] = None,
) -> dict:
    """
    Run exactly (warmup_steps + benchmark_steps) training steps for one experiment.

    The first warmup_steps steps are excluded from timing to avoid measuring:
      - CUDA kernel JIT compilation (first call to each new kernel)
      - DataLoader worker process spin-up (first few batches)
      - GPU cache warm-up effects

    Returns a dict of benchmark metrics for this experiment.
    """
    print(f"\n{'─' * 60}")
    print(f"Running: {spec.name}")
    print(f"  workers={spec.num_workers}  pin={spec.pin_memory}  "
          f"persist={spec.persistent_workers}  prefetch={spec.prefetch_factor}")
    print(f"  non_blocking={spec.non_blocking_h2d}  "
          f"amp={spec.use_amp} ({spec.amp_dtype})")

    set_seed(seed)

    # Build CLIPProcessor + tokenizer (shared across experiments)
    processor = build_clip_processor()
    tokenizer = build_caption_tokenizer()

    # Build DataLoader with experiment-specific knobs
    ann_file  = os.path.join(data_root, "annotations", "captions_train2017.json")
    image_dir = os.path.join(data_root, "train2017")
    loader    = create_coco_dataloader(
        ann_file=ann_file,
        image_dir=image_dir,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=spec.num_workers,
        pin_memory=spec.pin_memory,
        persistent_workers=spec.persistent_workers,
        prefetch_factor=spec.prefetch_factor,
        max_samples=max_samples,
        split="train",
    )

    # Build model (fresh for each experiment so no state leaks)
    model_cfg = VisionModelConfig()
    train_cfg = VisionTrainConfig(use_amp=spec.use_amp, amp_dtype=spec.amp_dtype)
    model     = NanoChatVisionModel(model_cfg).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=3e-4, betas=(0.9, 0.95),
    )

    autocast_ctx, scaler = build_amp_context(train_cfg, str(device))
    model.train()

    # ── Timing accumulators ───────────────────────────────────────────────────
    iter_times_ms:   List[float] = []
    dl_wait_times:   List[float] = []
    h2d_times:       List[float] = []
    forward_times:   List[float] = []
    backward_times:  List[float] = []
    peak_mem_mb = 0.0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    step       = 0
    total_steps = warmup_steps + benchmark_steps
    data_iter   = iter(loader)

    t_start = time.perf_counter()

    while step < total_steps:
        # Fetch a batch (cycle through the dataset if needed)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        t_dl_done = time.perf_counter()

        # ── H2D copy ─────────────────────────────────────────────────────────
        with CUDATimer(device) as h2d_t:
            pixel_values = batch["pixel_values"].to(device, non_blocking=spec.non_blocking_h2d)
            input_ids    = batch["input_ids"].to(device, non_blocking=spec.non_blocking_h2d)
            labels       = batch["labels"].to(device, non_blocking=spec.non_blocking_h2d)
            if spec.non_blocking_h2d and device.type == "cuda":
                torch.cuda.synchronize(device)

        # ── Forward ──────────────────────────────────────────────────────────
        with CUDATimer(device) as fwd_t:
            with autocast_ctx():
                out  = model(pixel_values=pixel_values, input_ids=input_ids, labels=labels)
                loss = out["loss"]

        # ── Backward ─────────────────────────────────────────────────────────
        with CUDATimer(device) as bwd_t:
            scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        t_iter_done = time.perf_counter()

        # Only record timing AFTER warmup
        if step >= warmup_steps:
            dl_wait_ms  = (t_dl_done - t_start) * 1000.0
            total_ms    = (t_iter_done - t_start) * 1000.0
            iter_times_ms.append(total_ms)
            dl_wait_times.append(dl_wait_ms)
            h2d_times.append(h2d_t.elapsed_ms)
            forward_times.append(fwd_t.elapsed_ms)
            backward_times.append(bwd_t.elapsed_ms)

        t_start = time.perf_counter()
        step   += 1

    # Peak GPU memory after all steps
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2

    # ── Aggregate statistics ──────────────────────────────────────────────────
    def _stats(vals: list) -> dict:
        if not vals:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0}
        sorted_v = sorted(vals)
        return {
            "mean":   statistics.mean(vals),
            "median": statistics.median(vals),
            "p95":    sorted_v[int(0.95 * len(sorted_v))],
        }

    iter_stats = _stats(iter_times_ms)
    mean_iter  = iter_stats["mean"]
    throughput = batch_size / max(mean_iter / 1000.0, 1e-9)

    result = {
        "experiment":          spec.name,
        "num_workers":         spec.num_workers,
        "pin_memory":          spec.pin_memory,
        "persistent_workers":  spec.persistent_workers,
        "prefetch_factor":     spec.prefetch_factor,
        "non_blocking_h2d":    spec.non_blocking_h2d,
        "use_amp":             spec.use_amp,
        "amp_dtype":           spec.amp_dtype,
        "batch_size":          batch_size,
        "steps_timed":         len(iter_times_ms),
        "throughput_sps":      round(throughput, 2),
        "iter_ms_mean":        round(iter_stats["mean"], 2),
        "iter_ms_median":      round(iter_stats["median"], 2),
        "iter_ms_p95":         round(iter_stats["p95"], 2),
        "dl_wait_ms_mean":     round(_stats(dl_wait_times)["mean"], 2),
        "dl_wait_ms_p95":      round(_stats(dl_wait_times)["p95"], 2),
        "h2d_ms_mean":         round(_stats(h2d_times)["mean"], 2),
        "forward_ms_mean":     round(_stats(forward_times)["mean"], 2),
        "backward_ms_mean":    round(_stats(backward_times)["mean"], 2),
        "peak_mem_mb":         round(peak_mem_mb, 1),
    }

    print(f"  → throughput={throughput:.1f} sps  "
          f"iter={mean_iter:.1f}ms  "
          f"dl_wait={_stats(dl_wait_times)['mean']:.1f}ms  "
          f"peak={peak_mem_mb:.0f}MB")

    # Clean up to free GPU memory before the next experiment
    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Inference latency benchmark
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def benchmark_inference(
    spec: ExperimentSpec,
    data_root: str,
    batch_sizes: List[int],
    max_new_tokens: int,
    warmup_steps: int,
    benchmark_steps: int,
    device: torch.device,
    seed: int,
) -> List[dict]:
    """
    Measure inference (caption generation) latency at multiple batch sizes.

    For each batch size we:
      1. Load a batch of real images from COCO val.
      2. Call model.generate() with visual token caching enabled.
      3. Time just the generate() call using CUDA Events.

    The visual KV cache (pre-computing CLIP tokens once before the decode loop)
    is the key inference optimisation in NanoChat-V.  This is always enabled here.

    Returns a list of result dicts, one per (experiment, batch_size) pair.
    """
    print(f"  Inference benchmark: {spec.name}")
    set_seed(seed)

    processor = build_clip_processor()
    tokenizer = build_caption_tokenizer()

    ann_file  = os.path.join(data_root, "annotations", "captions_val2017.json")
    image_dir = os.path.join(data_root, "val2017")

    model_cfg = VisionModelConfig()
    model     = NanoChatVisionModel(model_cfg).to(device)
    model.eval()

    results = []

    for bs in batch_sizes:
        # Load enough samples for (warmup_steps + benchmark_steps) batches
        needed = (warmup_steps + benchmark_steps) * bs
        loader = create_coco_dataloader(
            ann_file=ann_file,
            image_dir=image_dir,
            processor=processor,
            tokenizer=tokenizer,
            batch_size=bs,
            shuffle=False,
            num_workers=0,          # inference benchmark: keep it simple
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            max_samples=min(needed, 5000),
            split="val",
        )

        latencies_ms: List[float] = []
        step      = 0
        data_iter = iter(loader)

        while step < warmup_steps + benchmark_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch     = next(data_iter)

            pixel_values = batch["pixel_values"].to(device)

            # Time the full generation call with CUDATimer
            with CUDATimer(device) as gen_timer:
                _ = model.generate(
                    pixel_values=pixel_values,
                    max_new_tokens=max_new_tokens,
                    temperature=1.0,
                    top_k=50,
                    use_visual_kv_cache=True,
                )

            if step >= warmup_steps:
                latencies_ms.append(gen_timer.elapsed_ms)
            step += 1

        if not latencies_ms:
            continue

        sorted_lat  = sorted(latencies_ms)
        p50_ms      = sorted_lat[int(0.50 * len(sorted_lat))]
        p95_ms      = sorted_lat[int(0.95 * len(sorted_lat))]
        mean_ms     = statistics.mean(latencies_ms)
        tokens_sec  = (bs * max_new_tokens) / max(mean_ms / 1000.0, 1e-9)

        res = {
            "experiment":    spec.name,
            "batch_size":    bs,
            "max_new_tokens": max_new_tokens,
            "lat_mean_ms":   round(mean_ms, 2),
            "lat_p50_ms":    round(p50_ms, 2),
            "lat_p95_ms":    round(p95_ms, 2),
            "tokens_per_sec": round(tokens_sec, 1),
        }
        results.append(res)
        print(f"    bs={bs}  mean={mean_ms:.1f}ms  "
              f"p50={p50_ms:.1f}ms  p95={p95_ms:.1f}ms  "
              f"tokens/s={tokens_sec:.1f}")

    del model, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("NanoChat-V Benchmark Suite")
    print(f"  device={device}  steps={args.benchmark_steps}  "
          f"warmup={args.warmup_steps}  batch={args.batch_size}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU: {props.name}  SM={props.major}.{props.minor}  "
              f"VRAM={props.total_memory // 1024**2} MB")
    print("=" * 70)

    # Select which experiments to run
    experiments = EXPERIMENT_MATRIX
    if args.experiments:
        experiments = [e for e in EXPERIMENT_MATRIX if e.name in args.experiments]
        print(f"Running subset: {[e.name for e in experiments]}")

    # ── Training throughput benchmarks ────────────────────────────────────────
    print("\n[1] Training Throughput Benchmarks")
    training_results: List[dict] = []

    for spec in experiments:
        result = benchmark_training(
            spec=spec,
            data_root=args.data_root,
            batch_size=args.batch_size,
            benchmark_steps=args.benchmark_steps,
            warmup_steps=args.warmup_steps,
            device=device,
            seed=args.seed,
            max_samples=args.max_samples,
        )
        training_results.append(result)

    # Save training results
    train_csv  = os.path.join(args.output_dir, "benchmark_summary.csv")
    train_json = os.path.join(args.output_dir, "benchmark_summary.json")
    write_csv(train_csv, training_results)
    save_json(training_results, train_json)
    print(f"\nTraining results saved → {train_csv}")

    # ── Inference latency benchmarks ──────────────────────────────────────────
    if not args.skip_inference:
        print("\n[2] Inference Latency Benchmarks")
        inf_results: List[dict] = []

        # Run inference benchmark only for baseline + opt_all (most informative comparison)
        inf_specs = [e for e in experiments if e.name in ("A_baseline", "opt_all")]
        if not inf_specs:
            inf_specs = experiments[:1]   # at least one if user filtered

        for spec in inf_specs:
            res = benchmark_inference(
                spec=spec,
                data_root=args.data_root,
                batch_sizes=args.inf_batch_sizes,
                max_new_tokens=args.max_new_tokens,
                warmup_steps=args.warmup_steps,
                benchmark_steps=args.benchmark_steps,
                device=device,
                seed=args.seed,
            )
            inf_results.extend(res)

        inf_csv = os.path.join(args.output_dir, "inference_summary.csv")
        write_csv(inf_csv, inf_results)
        save_json(inf_results, os.path.join(args.output_dir, "inference_summary.json"))
        print(f"Inference results saved → {inf_csv}")

    # ── Speedup summary table ─────────────────────────────────────────────────
    _print_speedup_table(training_results)


def _print_speedup_table(results: List[dict]) -> None:
    """
    Print a human-readable speedup table relative to the baseline experiment (A).

    Shows for each experiment:
      - Throughput (samples/sec)
      - Speedup vs baseline (×)
      - Peak memory (MB)
      - Memory saving vs baseline (%)
    """
    if not results:
        return

    # Find baseline (experiment A or first entry)
    baseline = next((r for r in results if r["experiment"] == "A_baseline"), results[0])
    base_sps = baseline["throughput_sps"]
    base_mem = baseline["peak_mem_mb"]

    print("\n" + "=" * 70)
    print(f"{'Experiment':<28} {'Throughput':>12} {'Speedup':>9} "
          f"{'PeakMem':>10} {'MemSave':>8}")
    print("-" * 70)

    for r in results:
        sps     = r["throughput_sps"]
        mem     = r["peak_mem_mb"]
        speedup = sps / max(base_sps, 1e-9)
        mem_pct = 100.0 * (1.0 - mem / max(base_mem, 1e-9))
        print(f"  {r['experiment']:<26} {sps:>10.1f}  "
              f"  {speedup:>6.2f}×  "
              f"  {mem:>8.0f}MB  "
              f"  {mem_pct:>6.1f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()
