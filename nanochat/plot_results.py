"""
nanochat/vision/plot_results.py

Generate all paper-quality figures for the NanoChat-V experiment report.

Figures produced
----------------
1.  throughput_bar.png        — Training throughput (samples/sec) across experiments A-H + opt_all
2.  memory_bar.png            — Peak GPU memory (MB) per experiment
3.  latency_bar.png           — Inference latency (p50/p95 ms) at batch sizes [1, 4, 16]
4.  tokens_per_sec.png        — Inference tokens/sec vs batch size (line plot)
5.  dl_sweep_heatmap.png      — DataLoader wait (ms) heatmap: num_workers × batch_size
6.  speedup_waterfall.png     — Cumulative speedup waterfall: each opt adds one bar
7.  component_timing_pie.png  — Pie chart: % of forward pass time per component
8.  loss_curve.png            — Train/val loss curves (from train_log.csv)

Usage
-----
# Generate all figures from results/ directory:
python -m nanochat.vision.plot_results \
    --results_dir results \
    --output_dir results/figures

# Generate only specific figures:
python -m nanochat.vision.plot_results \
    --results_dir results \
    --figures throughput_bar speedup_waterfall

Each figure is saved as both PNG (300 dpi, for reports) and PDF (vector, for papers).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Matplotlib setup (use Agg backend so plotting works on headless GCP instances)
# ─────────────────────────────────────────────────────────────────────────────

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Use a clean, paper-quality style
plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi":   300,
    "axes.grid":    True,
    "grid.alpha":   0.3,
})

# Colour palette: colourblind-safe Set2
PALETTE = [
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
    "#1f78b4",
]

EXPERIMENT_ORDER = [
    "A_baseline",
    "B_workers8",
    "C_pin_memory",
    "D_persistent_workers",
    "E_prefetch",
    "F_non_blocking",
    "G_amp_fp16",
    "H_amp_bf16",
    "opt_all",
]
EXPERIMENT_LABELS = {
    "A_baseline":            "A: Baseline",
    "B_workers8":            "B: +Workers(8)",
    "C_pin_memory":          "C: +PinMemory",
    "D_persistent_workers":  "D: +PersistWorkers",
    "E_prefetch":            "E: +Prefetch(2)",
    "F_non_blocking":        "F: +NonBlocking",
    "G_amp_fp16":            "G: +AMP fp16",
    "H_amp_bf16":            "H: +AMP bf16",
    "opt_all":               "opt_all",
}


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate all NanoChat-V result figures",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results_dir", default="results",
                   help="Directory containing benchmark_summary.csv, etc.")
    p.add_argument("--output_dir", default="results/figures",
                   help="Directory to save generated figures")
    p.add_argument("--figures", nargs="*", default=None,
                   help="Subset of figures to generate (default: all 8)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_json(path: str) -> Optional[list]:
    """Load a JSON file; return None if the file doesn't exist."""
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def _load_csv(path: str) -> Optional[List[dict]]:
    """Load a CSV file as a list of dicts; return None if missing."""
    if not os.path.exists(path):
        print(f"  [WARN] Not found: {path}")
        return None
    import csv
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _sort_experiments(rows: List[dict], key: str = "experiment") -> List[dict]:
    """Sort experiment rows according to EXPERIMENT_ORDER."""
    order_map = {name: i for i, name in enumerate(EXPERIMENT_ORDER)}
    return sorted(rows, key=lambda r: order_map.get(r.get(key, ""), 999))


def _save_fig(fig: plt.Figure, output_dir: str, name: str) -> None:
    """Save a figure as both PNG (raster) and PDF (vector)."""
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=300)
    print(f"  Saved {name}.png + .pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Training throughput bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_throughput_bar(data: List[dict], output_dir: str) -> None:
    """
    Horizontal bar chart showing training throughput (samples/sec) for all 9 experiments.
    Experiments are ordered A → H → opt_all; baseline is grey, optimised is coloured.

    The x-axis uses a 0-base so the speedup is visually intuitive.
    """
    data = _sort_experiments(data)
    exps   = [EXPERIMENT_LABELS.get(r["experiment"], r["experiment"]) for r in data]
    sps    = [float(r["throughput_sps"]) for r in data]
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(data))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(exps, sps, color=colors, edgecolor="white", linewidth=0.5)

    # Annotate bars with value labels
    for bar, val in zip(bars, sps):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Training Throughput (samples / sec)")
    ax.set_title("Training Throughput by Experiment (higher is better)")
    ax.set_xlim(0, max(sps) * 1.18)
    ax.invert_yaxis()   # A at top, opt_all at bottom
    fig.tight_layout()
    _save_fig(fig, output_dir, "throughput_bar")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Peak GPU memory bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_memory_bar(data: List[dict], output_dir: str) -> None:
    """
    Horizontal bar chart showing peak GPU memory usage per experiment.
    AMP experiments (G, H) should show significantly lower memory than fp32 runs.
    """
    data = _sort_experiments(data)
    exps = [EXPERIMENT_LABELS.get(r["experiment"], r["experiment"]) for r in data]
    mems = [float(r["peak_mem_mb"]) for r in data]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = [PALETTE[i % len(PALETTE)] for i in range(len(data))]
    bars    = ax.barh(exps, mems, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, mems):
        ax.text(val + 10, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f} MB", va="center", ha="left", fontsize=9)

    ax.set_xlabel("Peak GPU Memory (MB)")
    ax.set_title("Peak GPU Memory by Experiment (lower is better)")
    ax.set_xlim(0, max(mems) * 1.15)
    ax.invert_yaxis()
    fig.tight_layout()
    _save_fig(fig, output_dir, "memory_bar")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Inference latency bar chart (p50 + p95)
# ─────────────────────────────────────────────────────────────────────────────

def plot_latency_bar(data: List[dict], output_dir: str) -> None:
    """
    Grouped bar chart: p50 and p95 inference latency (ms) for baseline vs opt_all
    at batch sizes 1, 4, 16.

    p50 = median latency (typical case)
    p95 = tail latency (worst-case for the 95th percentile of requests)
    """
    # Group by (experiment, batch_size)
    baseline = [r for r in data if r["experiment"] == "A_baseline"]
    opt_all  = [r for r in data if r["experiment"] == "opt_all"]

    batch_sizes = sorted(set(int(r["batch_size"]) for r in data))
    x           = np.arange(len(batch_sizes))
    width       = 0.2

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (rows, label, color) in enumerate([
        (baseline, "Baseline p50", PALETTE[0]),
        (baseline, "Baseline p95", PALETTE[1]),
        (opt_all,  "opt_all p50",  PALETTE[2]),
        (opt_all,  "opt_all p95",  PALETTE[3]),
    ]):
        offset = (i - 1.5) * width
        key    = "lat_p50_ms" if "p50" in label else "lat_p95_ms"
        vals   = []
        for bs in batch_sizes:
            row = next((r for r in rows if int(r["batch_size"]) == bs), None)
            vals.append(float(row[key]) if row else 0.0)
        ax.bar(x + offset, vals, width, label=label, color=color,
               edgecolor="white", linewidth=0.4)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Inference Latency: Baseline vs opt_all")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir, "latency_bar")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Tokens / second vs batch size (line plot)
# ─────────────────────────────────────────────────────────────────────────────

def plot_tokens_per_sec(data: List[dict], output_dir: str) -> None:
    """
    Line plot: inference tokens generated per second as batch size increases.

    Ideally tokens/sec should increase with batch size (GPU parallelism benefit).
    At some point VRAM limits the batch size — this plot shows where that is.
    """
    experiments = sorted(set(r["experiment"] for r in data))
    batch_sizes  = sorted(set(int(r["batch_size"]) for r in data))

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, exp in enumerate(experiments):
        rows  = [r for r in data if r["experiment"] == exp]
        bsizes = []
        tps    = []
        for bs in batch_sizes:
            row = next((r for r in rows if int(r["batch_size"]) == bs), None)
            if row:
                bsizes.append(bs)
                tps.append(float(row["tokens_per_sec"]))
        label = EXPERIMENT_LABELS.get(exp, exp)
        ax.plot(bsizes, tps, marker="o", label=label,
                color=PALETTE[i % len(PALETTE)], linewidth=2, markersize=6)

    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Tokens / Second")
    ax.set_title("Inference Throughput (tokens/sec) vs Batch Size")
    ax.legend()
    ax.set_xticks(batch_sizes)
    fig.tight_layout()
    _save_fig(fig, output_dir, "tokens_per_sec")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: DataLoader wait heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_dl_sweep_heatmap(data: List[dict], output_dir: str) -> None:
    """
    Heatmap: DataLoader batch wait time (ms) as a function of num_workers and batch_size.

    Rows = num_workers (0, 1, 2, 4, 8)
    Cols = batch_size  (4, 8, 16, 32)

    Bright (high) = slow data loading (bottleneck).
    Dark (low)    = fast data loading (pipeline not starved).

    This is a key insight figure: it shows how many workers are needed to saturate
    the GPU with the current disk and CPU bandwidth.
    """
    # Collect unique values
    all_workers = sorted(set(int(r["num_workers"]) for r in data))
    all_bs      = sorted(set(int(r["batch_size"])  for r in data))

    # Build matrix
    matrix = np.zeros((len(all_workers), len(all_bs)))
    for r in data:
        nw  = int(r["num_workers"])
        bs  = int(r["batch_size"])
        val = float(r["dl_wait_mean_ms"])
        ri  = all_workers.index(nw)
        ci  = all_bs.index(bs)
        matrix[ri, ci] = val

    fig, ax = plt.subplots(figsize=(8, 5))
    im      = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(all_bs)))
    ax.set_xticklabels(all_bs)
    ax.set_yticks(range(len(all_workers)))
    ax.set_yticklabels(all_workers)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("num_workers")
    ax.set_title("DataLoader Wait Time (ms) — lower is better")

    plt.colorbar(im, ax=ax, label="Wait time (ms)")

    # Annotate cells
    for i in range(len(all_workers)):
        for j in range(len(all_bs)):
            val = matrix[i, j]
            color = "white" if val > matrix.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=9, color=color)

    fig.tight_layout()
    _save_fig(fig, output_dir, "dl_sweep_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6: Speedup waterfall chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_speedup_waterfall(data: List[dict], output_dir: str) -> None:
    """
    Waterfall (cumulative) bar chart showing how each optimisation adds to the
    total speedup relative to the baseline.

    Each bar shows the MARGINAL speedup contribution of adding one optimisation.
    The final bar (opt_all) shows the total speedup of all optimisations combined.

    This is the most important figure for the report: it shows which optimisations
    give the biggest bang for the buck.
    """
    data = _sort_experiments(data)

    # Find baseline throughput
    baseline_row = next((r for r in data if r["experiment"] == "A_baseline"), data[0])
    base_sps     = float(baseline_row["throughput_sps"])

    labels   = []
    speedups = []
    for r in data:
        exp  = r["experiment"]
        sps  = float(r["throughput_sps"])
        spdup = sps / max(base_sps, 1e-9)
        labels.append(EXPERIMENT_LABELS.get(exp, exp))
        speedups.append(spdup)

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors  = [PALETTE[8] if "opt_all" in lab else PALETTE[i % 8]
               for i, lab in enumerate(labels)]

    bars = ax.bar(x, speedups, color=colors, edgecolor="white", linewidth=0.5)

    # Reference line at 1.0× (baseline)
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.0, label="Baseline (1×)")

    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                f"{val:.2f}×", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Speedup relative to Baseline (×)")
    ax.set_title("Training Speedup per Experiment (higher is better)")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, output_dir, "speedup_waterfall")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7: Component timing pie chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_component_timing_pie(data: dict, output_dir: str) -> None:
    """
    Pie chart showing what fraction of the FORWARD pass each component takes.

    Components:
      - CLIP forward      (frozen vision backbone — typically the largest chunk)
      - Projection MLP    (small but non-trivial)
      - GPT + cross-attn  (language backbone + 4 cross-attention layers)
      - Loss              (cross-entropy — usually tiny)

    This helps answer: "where should we focus optimisation efforts?"
    """
    keys = [
        ("clip_fwd_mean_ms",   "CLIP forward"),
        ("proj_mlp_mean_ms",   "Projection MLP"),
        ("gpt_xattn_mean_ms",  "GPT + Cross-Attn"),
        ("loss_mean_ms",       "Loss"),
    ]

    values = []
    labels = []
    for k, label in keys:
        v = float(data.get(k, 0.0))
        if v > 0:
            values.append(v)
            labels.append(label)

    if not values:
        print("  [WARN] No component timing data; skipping pie chart")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    wedge_colors = PALETTE[:len(values)]
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=90,
        pctdistance=0.75,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(10)

    total_ms = sum(values)
    ax.set_title(f"Forward Pass Time Breakdown\n(total ≈ {total_ms:.1f} ms per step)")
    fig.tight_layout()
    _save_fig(fig, output_dir, "component_timing_pie")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8: Training/validation loss curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_loss_curve(train_log: List[dict], val_log: Optional[List[dict]], output_dir: str) -> None:
    """
    Line plot: training loss and (optionally) validation loss vs global step.

    The training loss uses a rolling window average (from loss_avg column if present)
    to smooth the noisy per-step loss, making the trend clearer.

    The learning rate is shown on a secondary y-axis to visualise the warmup + cosine
    decay schedule against the loss trajectory.
    """
    steps     = [int(r["global_step"]) for r in train_log]
    train_loss = [float(r.get("loss_avg", r.get("loss", 0))) for r in train_log]
    lrs        = [float(r.get("lr", 0)) for r in train_log]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Training loss (left axis)
    ax1.plot(steps, train_loss, color=PALETTE[0], linewidth=1.5, label="Train loss (avg)")
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title("Training & Validation Loss Curve")

    # Validation loss (left axis, dashed)
    if val_log:
        val_steps = [int(r["global_step"]) for r in val_log]
        val_loss  = [float(r["val_loss"]) for r in val_log]
        ax1.plot(val_steps, val_loss, color=PALETTE[1], linewidth=2.0,
                 linestyle="--", marker="o", markersize=5, label="Val loss")

    ax1.legend(loc="upper right")

    # LR schedule (right axis)
    ax2 = ax1.twinx()
    ax2.plot(steps, lrs, color=PALETTE[6], linewidth=1.0, alpha=0.6, linestyle=":")
    ax2.set_ylabel("Learning Rate", color=PALETTE[6])
    ax2.tick_params(axis="y", labelcolor=PALETTE[6])

    fig.tight_layout()
    _save_fig(fig, output_dir, "loss_curve")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Which figures to generate
    all_figs = {
        "throughput_bar",
        "memory_bar",
        "latency_bar",
        "tokens_per_sec",
        "dl_sweep_heatmap",
        "speedup_waterfall",
        "component_timing_pie",
        "loss_curve",
    }
    selected = set(args.figures) if args.figures else all_figs
    results_dir = args.results_dir

    print("=" * 60)
    print("NanoChat-V: Generating Figures")
    print(f"  results_dir = {results_dir}")
    print(f"  output_dir  = {args.output_dir}")
    print("=" * 60)

    # ── Load data files ───────────────────────────────────────────────────────
    train_bench  = _load_json(os.path.join(results_dir, "benchmark_summary.json"))
    inf_bench    = _load_json(os.path.join(results_dir, "inference_summary.json"))
    dl_sweep     = _load_json(os.path.join(results_dir, "dataloader_sweep.json"))
    comp_timing  = _load_json(os.path.join(results_dir, "component_timing.json"))
    train_log    = _load_csv(os.path.join(results_dir, "train_log.csv"))
    val_log      = _load_csv(os.path.join(results_dir, "val_log.csv"))

    # ── Generate figures ──────────────────────────────────────────────────────
    if "throughput_bar" in selected and train_bench:
        print("\n[1] throughput_bar")
        plot_throughput_bar(train_bench, args.output_dir)

    if "memory_bar" in selected and train_bench:
        print("\n[2] memory_bar")
        plot_memory_bar(train_bench, args.output_dir)

    if "latency_bar" in selected and inf_bench:
        print("\n[3] latency_bar")
        plot_latency_bar(inf_bench, args.output_dir)

    if "tokens_per_sec" in selected and inf_bench:
        print("\n[4] tokens_per_sec")
        plot_tokens_per_sec(inf_bench, args.output_dir)

    if "dl_sweep_heatmap" in selected and dl_sweep:
        print("\n[5] dl_sweep_heatmap")
        plot_dl_sweep_heatmap(dl_sweep, args.output_dir)

    if "speedup_waterfall" in selected and train_bench:
        print("\n[6] speedup_waterfall")
        plot_speedup_waterfall(train_bench, args.output_dir)

    if "component_timing_pie" in selected and comp_timing:
        print("\n[7] component_timing_pie")
        # comp_timing is a single dict, not a list
        if isinstance(comp_timing, list):
            comp_timing = comp_timing[0]
        plot_component_timing_pie(comp_timing, args.output_dir)

    if "loss_curve" in selected and train_log:
        print("\n[8] loss_curve")
        plot_loss_curve(train_log, val_log, args.output_dir)

    print(f"\nAll figures saved to: {args.output_dir}/")
    print("  (PNG at 300 dpi + PDF vector format)")


if __name__ == "__main__":
    main()
