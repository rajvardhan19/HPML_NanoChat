"""
nanochat/vision/profile_vision.py

Detailed per-component CUDA timing and DataLoader sweep profiler for NanoChat-V.

What this script measures
-------------------------
1.  DataLoader sweep  — measures dataloader_wait_ms at each (num_workers, batch_size)
    combination so we can find the sweet-spot for our I/O pipeline.

2.  Per-component CUDA timing — breaks down one training step into:
      a. CLIP forward          (vision_encoder)
      b. Projection MLP        (vision_projection)
      c. GPT forward + XA      (forward_with_cross_attn — includes cross-attention)
      d. Loss computation      (F.cross_entropy)
      e. Backward pass         (loss.backward)

3.  PyTorch Profiler trace — writes TensorBoard-compatible JSON to profile_output/
    so you can open chrome://tracing or TensorBoard to see GPU kernel timings.

4.  Memory breakdown — VRAM used by each component (frozen CLIP vs trainable head).

Usage
-----
# Run the full profiling suite:
python -m nanochat.vision.profile_vision \
    --data_root data/coco \
    --output_dir results/profiling

# DataLoader sweep only (no GPU needed):
python -m nanochat.vision.profile_vision \
    --data_root data/coco \
    --output_dir results/profiling \
    --skip_model true \
    --num_workers_list 0 2 4 8 \
    --batch_sizes 4 8 16

# Component timing only (use existing data):
python -m nanochat.vision.profile_vision \
    --data_root data/coco \
    --output_dir results/profiling \
    --skip_dataloader true \
    --profile_steps 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from typing import List, Optional

import torch
import torch.nn as nn

from nanochat.vision.config import VisionModelConfig, VisionTrainConfig
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
    get_component_memory_mb,
    count_parameters,
    write_csv,
    save_json,
    bool_arg,
)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile NanoChat-V DataLoader + per-component CUDA timing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data_root", default="data/coco")
    p.add_argument("--output_dir", default="results/profiling")
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_samples", type=int, default=2000,
                   help="Cap dataset to limit disk I/O during profiling")

    # DataLoader sweep
    p.add_argument("--skip_dataloader", type=bool_arg, default=False)
    p.add_argument("--num_workers_list", nargs="+", type=int, default=[0, 1, 2, 4, 8])
    p.add_argument("--batch_sizes", nargs="+", type=int, default=[4, 8, 16, 32])
    p.add_argument("--dl_sweep_steps", type=int, default=50,
                   help="Number of batches to time per (workers, batch_size) combo")

    # Component timing
    p.add_argument("--skip_model", type=bool_arg, default=False)
    p.add_argument("--profile_steps", type=int, default=20,
                   help="Number of steps for per-component timing")
    p.add_argument("--warmup_steps", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for component timing")

    # PyTorch Profiler
    p.add_argument("--run_profiler", type=bool_arg, default=True,
                   help="Run torch.profiler.profile and save TensorBoard trace")
    p.add_argument("--profiler_steps", type=int, default=5,
                   help="Steps to capture in the PyTorch Profiler trace")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_dataloader_sweep(
    data_root: str,
    num_workers_list: List[int],
    batch_sizes: List[int],
    sweep_steps: int,
    max_samples: Optional[int],
    output_dir: str,
) -> List[dict]:
    """
    Measure DataLoader throughput and wait latency for every combination of
    (num_workers, batch_size).

    Why this matters
    ----------------
    The DataLoader wait time is the bottleneck for experiment A (baseline).
    With num_workers=0 (single-process loading), the GPU stalls waiting for the
    next batch while the CPU decodes and preprocesses images.

    With num_workers>0, worker subprocesses prefetch and preprocess concurrently
    with GPU compute, overlapping disk I/O with training forward/backward.

    We measure:
      - Batch latency: wall time for the DataLoader to yield one batch
      - Throughput: images/second delivered by the loader
    """
    print("\n[DataLoader Sweep]")
    print(f"  num_workers_list: {num_workers_list}")
    print(f"  batch_sizes:      {batch_sizes}")

    processor = build_clip_processor()
    tokenizer = build_caption_tokenizer()
    ann_file  = os.path.join(data_root, "annotations", "captions_train2017.json")
    image_dir = os.path.join(data_root, "train2017")

    results = []

    for nw in num_workers_list:
        for bs in batch_sizes:
            print(f"  num_workers={nw}  batch_size={bs} …", end="", flush=True)

            loader = create_coco_dataloader(
                ann_file=ann_file,
                image_dir=image_dir,
                processor=processor,
                tokenizer=tokenizer,
                batch_size=bs,
                shuffle=True,
                num_workers=nw,
                pin_memory=nw > 0,          # pin_memory only useful with workers
                persistent_workers=nw > 0,
                prefetch_factor=2 if nw > 0 else None,
                max_samples=max_samples,
                split="train",
            )

            latencies_ms: List[float] = []
            step = 0
            data_iter = iter(loader)

            t_start = time.perf_counter()
            while step < sweep_steps:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(loader)
                    batch     = next(data_iter)
                    t_start   = time.perf_counter()  # reset after cycling
                    continue

                t_end   = time.perf_counter()
                elapsed = (t_end - t_start) * 1000.0
                if step >= 3:   # skip first 3 (worker spinup)
                    latencies_ms.append(elapsed)
                t_start = time.perf_counter()
                step   += 1

            if not latencies_ms:
                continue

            mean_ms = statistics.mean(latencies_ms)
            p95_ms  = sorted(latencies_ms)[int(0.95 * len(latencies_ms))]
            imgs_s  = bs / max(mean_ms / 1000.0, 1e-9)

            res = {
                "num_workers":     nw,
                "batch_size":      bs,
                "dl_wait_mean_ms": round(mean_ms, 2),
                "dl_wait_p95_ms":  round(p95_ms, 2),
                "images_per_sec":  round(imgs_s, 1),
            }
            results.append(res)
            print(f" → {mean_ms:.1f}ms  {imgs_s:.0f} imgs/s")

            del loader

    # Save sweep results
    os.makedirs(output_dir, exist_ok=True)
    csv_path  = os.path.join(output_dir, "dataloader_sweep.csv")
    json_path = os.path.join(output_dir, "dataloader_sweep.json")
    write_csv(csv_path, results)
    save_json(results, json_path)
    print(f"\n  Saved → {csv_path}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-component CUDA timing
# ─────────────────────────────────────────────────────────────────────────────

def run_component_timing(
    data_root: str,
    batch_size: int,
    profile_steps: int,
    warmup_steps: int,
    device: torch.device,
    output_dir: str,
    run_profiler: bool,
    profiler_steps: int,
    max_samples: Optional[int],
) -> dict:
    """
    Time each component of the NanoChat-V forward + backward pass separately.

    Components timed
    ----------------
    a) CLIP forward:         pixel_values → (B, 197, 768) visual tokens
       (This is the frozen vision encoder — normally the biggest single component)

    b) Projection MLP:       visual tokens → projected_visual
       (Small 2-layer MLP; should be fast)

    c) GPT forward + cross-attn:  input_ids + visual tokens → logits
       (Language backbone — includes 4× cross-attention modules at layers 8-11)

    d) Loss:                 logits vs labels → scalar
       (F.cross_entropy; fast)

    e) Backward:             loss.backward() over all trainable params
       (Proportional to forward pass; GPT backward + projection backward)

    We use CUDATimer (torch.cuda.Event) for accurate GPU timing.
    CPU timer would include Python overhead; Event timing measures actual GPU ops.
    """
    print(f"\n[Component Timing]  batch_size={batch_size}  steps={profile_steps}")

    set_seed(42)
    processor = build_clip_processor()
    tokenizer = build_caption_tokenizer()

    ann_file  = os.path.join(data_root, "annotations", "captions_train2017.json")
    image_dir = os.path.join(data_root, "train2017")

    loader = create_coco_dataloader(
        ann_file=ann_file,
        image_dir=image_dir,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        prefetch_factor=2,
        max_samples=max_samples,
        split="train",
    )

    # Build model
    model_cfg = VisionModelConfig()
    model     = NanoChatVisionModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    model.train()

    # ── PyTorch profiler context ──────────────────────────────────────────────
    profiler = None
    if run_profiler:
        trace_dir = os.path.join(output_dir, "profiler_trace")
        os.makedirs(trace_dir, exist_ok=True)
        profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=warmup_steps,
                warmup=1,
                active=profiler_steps,
                repeat=1,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            record_shapes=True,
            profile_memory=True,
        )
        profiler.start()

    # Timing accumulators
    clip_times:   List[float] = []
    proj_times:   List[float] = []
    gpt_times:    List[float] = []
    loss_times:   List[float] = []
    bwd_times:    List[float] = []

    step      = 0
    data_iter = iter(loader)
    total_needed = warmup_steps + profile_steps

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while step < total_needed:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch     = next(data_iter)

        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids    = batch["input_ids"].to(device, non_blocking=True)
        labels       = batch["labels"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        # ── a) CLIP forward ───────────────────────────────────────────────────
        with CUDATimer(device) as clip_t:
            # Replicate exactly what NanoChatVisionModel.forward() does:
            # vision_encoder runs under no_grad since CLIP is frozen
            visual_tokens = model.vision_encoder(pixel_values)

        # ── b) VisionProjection MLP ───────────────────────────────────────────
        with CUDATimer(device) as proj_t:
            projected = model.vision_projection(visual_tokens)

        # ── c) GPT forward + cross-attention ──────────────────────────────────
        with CUDATimer(device) as gpt_t:
            logits = model.gpt.forward_with_cross_attn(
                idx=input_ids,
                visual_tokens=projected,
                cross_attn_layers=model.cross_attn_layer_indices,
                cross_attn_modules=model.cross_attn_modules,
            )

        # ── d) Loss ───────────────────────────────────────────────────────────
        import torch.nn.functional as F
        with CUDATimer(device) as loss_t:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        # ── e) Backward ───────────────────────────────────────────────────────
        with CUDATimer(device) as bwd_t:
            loss.backward()

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if profiler is not None:
            profiler.step()

        # Record only after warmup
        if step >= warmup_steps:
            clip_times.append(clip_t.elapsed_ms)
            proj_times.append(proj_t.elapsed_ms)
            gpt_times.append(gpt_t.elapsed_ms)
            loss_times.append(loss_t.elapsed_ms)
            bwd_times.append(bwd_t.elapsed_ms)

        step += 1

    if profiler is not None:
        profiler.stop()
        print(f"  Profiler trace saved → {trace_dir}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    def _mean(vals): return round(statistics.mean(vals), 2) if vals else 0.0
    def _p95(vals):
        if not vals: return 0.0
        return round(sorted(vals)[int(0.95 * len(vals))], 2)

    # Total forward (sum of a+b+c+d)
    total_fwd  = [c+p+g+l for c,p,g,l in zip(clip_times, proj_times, gpt_times, loss_times)]
    total_iter = [f+b for f,b in zip(total_fwd, bwd_times)]

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0

    results = {
        "batch_size":          batch_size,
        "steps_timed":         len(clip_times),
        "clip_fwd_mean_ms":    _mean(clip_times),
        "clip_fwd_p95_ms":     _p95(clip_times),
        "proj_mlp_mean_ms":    _mean(proj_times),
        "gpt_xattn_mean_ms":   _mean(gpt_times),
        "loss_mean_ms":        _mean(loss_times),
        "backward_mean_ms":    _mean(bwd_times),
        "total_fwd_mean_ms":   _mean(total_fwd),
        "total_iter_mean_ms":  _mean(total_iter),
        "throughput_sps":      round(batch_size / max(_mean(total_iter) / 1000.0, 1e-9), 1),
        "peak_mem_mb":         round(peak_mem, 1),
    }

    # Print breakdown
    fwd_total = _mean(total_fwd)
    print(f"\n  Component timing (mean over {len(clip_times)} steps):")
    print(f"    CLIP forward:       {results['clip_fwd_mean_ms']:7.2f} ms  "
          f"({100*results['clip_fwd_mean_ms']/max(fwd_total,1e-3):.0f}% of fwd)")
    print(f"    Projection MLP:     {results['proj_mlp_mean_ms']:7.2f} ms  "
          f"({100*results['proj_mlp_mean_ms']/max(fwd_total,1e-3):.0f}% of fwd)")
    print(f"    GPT + cross-attn:   {results['gpt_xattn_mean_ms']:7.2f} ms  "
          f"({100*results['gpt_xattn_mean_ms']/max(fwd_total,1e-3):.0f}% of fwd)")
    print(f"    Loss:               {results['loss_mean_ms']:7.2f} ms")
    print(f"    Backward:           {results['backward_mean_ms']:7.2f} ms")
    print(f"    ── Total fwd:       {_mean(total_fwd):7.2f} ms")
    print(f"    ── Total iter:      {_mean(total_iter):7.2f} ms")
    print(f"    Throughput:         {results['throughput_sps']} sps")
    print(f"    Peak GPU memory:    {peak_mem:.0f} MB")

    # ── Memory breakdown by component ─────────────────────────────────────────
    comp_memory = {
        "vision_encoder_mb":    get_component_memory_mb(model.vision_encoder),
        "vision_projection_mb": get_component_memory_mb(model.vision_projection),
        "cross_attn_mb":        get_component_memory_mb(model.cross_attn_modules),
        "gpt_mb":               get_component_memory_mb(model.gpt),
    }
    results.update(comp_memory)
    print(f"\n  Parameter memory (FP32 equivalent):")
    for k, v in comp_memory.items():
        print(f"    {k:<28} {v:7.1f} MB")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    save_json(results, os.path.join(output_dir, "component_timing.json"))
    print(f"\n  Saved → {output_dir}/component_timing.json")

    del model, optimizer, loader
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("NanoChat-V Profiling Suite")
    print(f"  device={device}  output={args.output_dir}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        print(f"  GPU: {props.name}  SM={props.major}.{props.minor}  "
              f"VRAM={props.total_memory // 1024**2} MB")
    print("=" * 70)

    dl_results   = None
    comp_results = None

    # ── DataLoader sweep ──────────────────────────────────────────────────────
    if not args.skip_dataloader:
        dl_results = run_dataloader_sweep(
            data_root=args.data_root,
            num_workers_list=args.num_workers_list,
            batch_sizes=args.batch_sizes,
            sweep_steps=args.dl_sweep_steps,
            max_samples=args.max_samples,
            output_dir=args.output_dir,
        )

    # ── Component timing ──────────────────────────────────────────────────────
    if not args.skip_model:
        comp_results = run_component_timing(
            data_root=args.data_root,
            batch_size=args.batch_size,
            profile_steps=args.profile_steps,
            warmup_steps=args.warmup_steps,
            device=device,
            output_dir=args.output_dir,
            run_profiler=args.run_profiler,
            profiler_steps=args.profiler_steps,
            max_samples=args.max_samples,
        )

    print(f"\nProfiling complete.  Results in: {args.output_dir}/")
    if args.run_profiler and not args.skip_model:
        print("  → Open TensorBoard to view CUDA kernel trace:")
        print(f"    tensorboard --logdir {os.path.join(args.output_dir, 'profiler_trace')}")


if __name__ == "__main__":
    main()
