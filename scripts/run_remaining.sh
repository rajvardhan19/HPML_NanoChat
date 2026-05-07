#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs results

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "=== Tier-2: A-H Benchmarks ==="
run_exp() {
    local NAME="$1"; shift
    log "  Running: $NAME"
    python3 -m nanochat.vision.train_vision \
        --experiment_name "$NAME" --output_dir checkpoints \
        --data_root data/coco --batch_size 8 \
        --benchmark_only true --benchmark_steps 100 \
        "$@" 2>&1 | tee "logs/tier2_${NAME}.log"
}

run_exp A_baseline --num_workers 0 --pin_memory false --persistent_workers false --use_amp false
run_exp B_workers8 --num_workers 8 --pin_memory false --persistent_workers false --use_amp false
run_exp C_pin_memory --num_workers 0 --pin_memory true --persistent_workers false --use_amp false
run_exp D_persistent_workers --num_workers 4 --pin_memory false --persistent_workers true --use_amp false
run_exp E_prefetch --num_workers 4 --prefetch_factor 2 --use_amp false
run_exp F_non_blocking --num_workers 0 --pin_memory true --non_blocking_h2d true --use_amp false
run_exp G_amp_fp16 --use_amp true --amp_dtype fp16
run_exp H_amp_bf16 --use_amp true --amp_dtype bf16

log "=== Benchmark Suite ==="
python3 -m nanochat.vision.benchmark_vision \
    --data_root data/coco --output_dir results \
    --benchmark_steps 100 --warmup_steps 10 --batch_size 8 \
    2>&1 | tee logs/benchmark.log

log "=== Evaluation (BLEU + CIDEr) ==="
python3 -m nanochat.vision.evaluate_captioning \
    --checkpoint checkpoints/opt_all/model.pt \
    --data_root data/coco --output_dir results \
    --batch_size 16 --num_workers 4 \
    --use_amp true --amp_dtype bf16 \
    2>&1 | tee logs/eval.log

log "=== Profiling ==="
python3 -m nanochat.vision.profile_vision \
    --data_root data/coco --output_dir results/profiling \
    --batch_size 8 --profile_steps 20 --warmup_steps 5 \
    --num_workers_list 0 1 2 4 8 --batch_sizes 4 8 16 32 \
    --run_profiler true --profiler_steps 5 \
    2>&1 | tee logs/profiling.log

log "=== Generating Plots ==="
python3 -m nanochat.vision.plot_results \
    --results_dir results --output_dir results/figures \
    2>&1 | tee logs/plots.log

log "=== All done! ==="
