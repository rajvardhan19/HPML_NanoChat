#!/usr/bin/env bash
# scripts/run_experiments.sh
#
# Orchestrate the full NanoChat-V experiment matrix.
#
# Runs (in order):
#   1. Tier-1: opt_all full training (1 epoch on full COCO train)
#   2. Tier-2: 100-step benchmarks for experiments A through H
#   3. Full benchmark suite (train throughput + inference latency)
#   4. BLEU + CIDEr evaluation on val set
#   5. DataLoader profiling sweep
#   6. Per-component CUDA timing
#   7. Plot all 8 figures
#
# GCP / L4 usage
# --------------
#   # SSH into your L4 instance via IAM tunnel, then:
#   cd ~/HPML_Project
#   conda activate nanochat     # or source venv/bin/activate
#   bash scripts/run_experiments.sh 2>&1 | tee logs/run_all.log
#
# Adjust BATCH_SIZE and NUM_WORKERS for your GPU VRAM and CPU core count.
# L4 has 24 GB VRAM — batch_size=16 in bf16 uses ~8 GB.

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_ROOT="${DATA_ROOT:-data/coco}"
CKPT_ROOT="${CKPT_ROOT:-checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"
LOG_DIR="${LOG_DIR:-logs}"

BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
SEED="${SEED:-42}"

PYTHON="${PYTHON:-python}"

mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ─── Tier-1: opt_all full training ────────────────────────────────────────────
log "=== TIER-1: Full Training (opt_all) ==="
${PYTHON} -m nanochat.vision.train_vision \
  --experiment_name opt_all \
  --output_dir "${CKPT_ROOT}" \
  --data_root "${DATA_ROOT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${NUM_EPOCHS}" \
  --num_workers 8 \
  --pin_memory true \
  --persistent_workers true \
  --prefetch_factor 2 \
  --non_blocking_h2d true \
  --use_amp true \
  --amp_dtype bf16 \
  --compile_model false \
  --use_wandb false \
  --log_interval 50 \
  --seed "${SEED}" \
  2>&1 | tee "${LOG_DIR}/tier1_opt_all.log"

log "Tier-1 opt_all complete"

# ─── Tier-2: 100-step benchmarks for experiments A-H ─────────────────────────
log "=== TIER-2: 100-step Benchmark Experiments ==="

run_exp() {
  local NAME="$1"; shift
  log "  Running: ${NAME}"
  ${PYTHON} -m nanochat.vision.train_vision \
    --experiment_name "${NAME}" \
    --output_dir "${CKPT_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --benchmark_only true \
    --benchmark_steps 100 \
    --seed "${SEED}" \
    "$@" \
    2>&1 | tee "${LOG_DIR}/tier2_${NAME}.log"
}

# A: Baseline — no optimisations
run_exp "A_baseline" \
  --num_workers 0 --pin_memory false --persistent_workers false \
  --use_amp false --compile_model false

# B: + num_workers=8
run_exp "B_workers8" \
  --num_workers 8 --pin_memory false --persistent_workers false \
  --use_amp false --compile_model false

# C: + pin_memory
run_exp "C_pin_memory" \
  --num_workers 0 --pin_memory true --persistent_workers false \
  --use_amp false --compile_model false

# D: + persistent_workers (requires num_workers > 0)
run_exp "D_persistent_workers" \
  --num_workers 4 --pin_memory false --persistent_workers true \
  --use_amp false --compile_model false

# E: + prefetch_factor=2 (requires num_workers > 0)
run_exp "E_prefetch" \
  --num_workers 4 --pin_memory false --persistent_workers false \
  --prefetch_factor 2 --use_amp false --compile_model false

# F: + non_blocking H2D transfer (most useful with pin_memory)
run_exp "F_non_blocking" \
  --num_workers 0 --pin_memory true --persistent_workers false \
  --non_blocking_h2d true --use_amp false --compile_model false

# G: + AMP fp16 (+ GradScaler)
run_exp "G_amp_fp16" \
  --num_workers 0 --pin_memory false --persistent_workers false \
  --use_amp true --amp_dtype fp16 --compile_model false

# H: + AMP bf16 (no GradScaler, stable on L4/Ampere+)
run_exp "H_amp_bf16" \
  --num_workers 0 --pin_memory false --persistent_workers false \
  --use_amp true --amp_dtype bf16 --compile_model false

log "Tier-2 benchmarks complete"

# ─── Full benchmark suite ─────────────────────────────────────────────────────
log "=== Running Full Benchmark Suite ==="
${PYTHON} -m nanochat.vision.benchmark_vision \
  --data_root "${DATA_ROOT}" \
  --output_dir "${RESULTS_DIR}" \
  --benchmark_steps 100 \
  --warmup_steps 10 \
  --batch_size "${BATCH_SIZE}" \
  2>&1 | tee "${LOG_DIR}/benchmark.log"

log "Benchmark suite complete → ${RESULTS_DIR}/benchmark_summary.csv"

# ─── BLEU + CIDEr evaluation ──────────────────────────────────────────────────
log "=== Running Caption Evaluation (BLEU + CIDEr) ==="
BEST_CKPT="${CKPT_ROOT}/opt_all/model.pt"
if [ -f "${BEST_CKPT}" ]; then
  ${PYTHON} -m nanochat.vision.evaluate_captioning \
    --checkpoint "${BEST_CKPT}" \
    --data_root "${DATA_ROOT}" \
    --output_dir "${RESULTS_DIR}" \
    --batch_size 16 \
    --num_workers 4 \
    --use_amp true \
    --amp_dtype bf16 \
    2>&1 | tee "${LOG_DIR}/eval.log"
  log "Evaluation complete → ${RESULTS_DIR}/eval_results.json"
else
  log "WARNING: checkpoint not found at ${BEST_CKPT}; skipping evaluation"
fi

# ─── DataLoader profiling ─────────────────────────────────────────────────────
log "=== Running DataLoader Sweep + Component Profiling ==="
${PYTHON} -m nanochat.vision.profile_vision \
  --data_root "${DATA_ROOT}" \
  --output_dir "${RESULTS_DIR}/profiling" \
  --batch_size "${BATCH_SIZE}" \
  --profile_steps 20 \
  --warmup_steps 5 \
  --num_workers_list 0 1 2 4 8 \
  --batch_sizes 4 8 16 32 \
  --run_profiler true \
  --profiler_steps 5 \
  2>&1 | tee "${LOG_DIR}/profiling.log"

log "Profiling complete → ${RESULTS_DIR}/profiling/"

# ─── Figures ──────────────────────────────────────────────────────────────────
log "=== Generating Result Figures ==="
${PYTHON} -m nanochat.vision.plot_results \
  --results_dir "${RESULTS_DIR}" \
  --output_dir "${RESULTS_DIR}/figures" \
  2>&1 | tee "${LOG_DIR}/plots.log"

log "Figures saved → ${RESULTS_DIR}/figures/"

# ─── Done ─────────────────────────────────────────────────────────────────────
log "=== All experiments complete ==="
echo ""
echo "Output summary:"
echo "  Checkpoints  : ${CKPT_ROOT}/"
echo "  Results CSV  : ${RESULTS_DIR}/benchmark_summary.csv"
echo "  Eval JSON    : ${RESULTS_DIR}/eval_results.json"
echo "  Figures      : ${RESULTS_DIR}/figures/"
echo "  Logs         : ${LOG_DIR}/"
echo ""
echo "To view profiler traces:"
echo "  tensorboard --logdir ${RESULTS_DIR}/profiling/profiler_trace"
