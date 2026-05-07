"""
scripts/experiment_flags.py

Print the exact CLI flags for each of the 9 experiments (A–H + opt_all).

Useful for copying individual experiment commands when running experiments
manually rather than through run_experiments.sh.

Usage
-----
    python scripts/experiment_flags.py                  # print all
    python scripts/experiment_flags.py A_baseline opt_all  # print subset
    python scripts/experiment_flags.py --format shell   # one-liner shell form
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ExperimentFlags:
    name: str
    num_workers: int      = 0
    pin_memory: bool      = False
    persistent_workers: bool = False
    prefetch_factor: Optional[int] = None
    non_blocking_h2d: bool = False
    use_amp: bool         = False
    amp_dtype: str        = "fp16"
    description: str      = ""


EXPERIMENTS: List[ExperimentFlags] = [
    ExperimentFlags(
        name="A_baseline",
        description="No optimisations — pure baseline",
    ),
    ExperimentFlags(
        name="B_workers8",
        num_workers=8,
        description="+ num_workers=8: parallel data loading",
    ),
    ExperimentFlags(
        name="C_pin_memory",
        pin_memory=True,
        description="+ pin_memory=True: page-locked host buffers for faster H2D",
    ),
    ExperimentFlags(
        name="D_persistent_workers",
        num_workers=4,
        persistent_workers=True,
        description="+ persistent_workers=True: avoid worker respawn between epochs",
    ),
    ExperimentFlags(
        name="E_prefetch",
        num_workers=4,
        prefetch_factor=2,
        description="+ prefetch_factor=2: each worker pre-fetches 2 batches ahead",
    ),
    ExperimentFlags(
        name="F_non_blocking",
        pin_memory=True,
        non_blocking_h2d=True,
        description="+ non_blocking H2D: overlap CPU→GPU copy with GPU compute",
    ),
    ExperimentFlags(
        name="G_amp_fp16",
        use_amp=True,
        amp_dtype="fp16",
        description="+ AMP fp16: autocast + GradScaler (works on all CUDA GPUs)",
    ),
    ExperimentFlags(
        name="H_amp_bf16",
        use_amp=True,
        amp_dtype="bf16",
        description="+ AMP bf16: autocast without GradScaler (Ampere+ only, e.g. L4)",
    ),
    ExperimentFlags(
        name="opt_all",
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        non_blocking_h2d=True,
        use_amp=True,
        amp_dtype="bf16",
        description="All optimisations: B+C+D+E+F+H combined",
    ),
]


def flags_to_str(exp: ExperimentFlags, mode: str = "multiline") -> str:
    """Format an ExperimentFlags as a CLI argument string."""
    args = [
        f"--experiment_name {exp.name}",
        f"--num_workers {exp.num_workers}",
        f"--pin_memory {'true' if exp.pin_memory else 'false'}",
        f"--persistent_workers {'true' if exp.persistent_workers else 'false'}",
    ]
    if exp.prefetch_factor is not None:
        args.append(f"--prefetch_factor {exp.prefetch_factor}")
    args += [
        f"--non_blocking_h2d {'true' if exp.non_blocking_h2d else 'false'}",
        f"--use_amp {'true' if exp.use_amp else 'false'}",
        f"--amp_dtype {exp.amp_dtype}",
    ]

    if mode == "shell":
        return " \\\n    ".join(args)
    else:
        return "\n".join(f"  {a}" for a in args)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Print CLI flags for each NanoChat-V experiment"
    )
    p.add_argument("experiments", nargs="*",
                   help="Names to print (default: all)")
    p.add_argument("--format", choices=["multiline", "shell"], default="multiline")
    args = p.parse_args()

    selected = [e for e in EXPERIMENTS
                if not args.experiments or e.name in args.experiments]

    if not selected:
        print(f"No experiments matched: {args.experiments}")
        print(f"Available: {[e.name for e in EXPERIMENTS]}")
        sys.exit(1)

    cmd_base = "python -m nanochat.vision.train_vision \\\n    "

    for exp in selected:
        print(f"\n{'='*60}")
        print(f"Experiment: {exp.name}")
        print(f"  {exp.description}")
        print(f"{'='*60}")
        if args.format == "shell":
            print(cmd_base + flags_to_str(exp, "shell") +
                  " \\\n    --benchmark_only true --benchmark_steps 100")
        else:
            print(flags_to_str(exp, "multiline"))


if __name__ == "__main__":
    main()
