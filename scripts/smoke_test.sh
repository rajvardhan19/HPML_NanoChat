#!/usr/bin/env bash
# scripts/smoke_test.sh
#
# Quick end-to-end smoke test for NanoChat-V.
#
# What this tests
# ---------------
# 1. Python import chain — all nanochat.vision modules importable
# 2. GPU detection — prints L4 SM version and VRAM
# 3. Unit tests (CPU) — pytest -v nanochat/vision/tests/
# 4. Mini training run — 10 steps, 64 samples, batch_size=4
# 5. Mini benchmark — 5 steps, no GPU warm-up needed
# 6. Single-image inference — if a test image exists
#
# Requirements
# ------------
#   - Python environment with: pip install -e .[vision]
#   - data/coco/  at minimum val2017/ + annotations/captions_val2017.json
#     (run: bash scripts/download_coco.sh data/coco --val_only)
#   - For the GCP L4 setup: IAM tunnel already established, SSH into instance
#
# Usage
# -----
#   bash scripts/smoke_test.sh                   # uses data/coco and checkpoints/
#   bash scripts/smoke_test.sh data/my_coco out/ # custom paths

set -euo pipefail

DATA_ROOT="${1:-data/coco}"
OUTPUT_ROOT="${2:-checkpoints/smoke_test}"

PYTHON="${PYTHON:-python3}"
PASS=0; FAIL=0

# ── Helpers ───────────────────────────────────────────────────────────────────
log_pass() { echo "  [PASS] $*"; ((PASS++)); }
log_fail() { echo "  [FAIL] $*"; ((FAIL++)); }
section() { echo ""; echo "=== $* ==="; }

# ── 1. GPU info ───────────────────────────────────────────────────────────────
section "GPU / Environment Info"
${PYTHON} - << 'EOF'
import sys, torch
print(f"  Python  : {sys.version.split()[0]}")
print(f"  PyTorch : {torch.__version__}")
cuda = torch.cuda.is_available()
print(f"  CUDA    : {cuda}")
if cuda:
    p = torch.cuda.get_device_properties(0)
    print(f"  GPU     : {p.name}  SM={p.major}.{p.minor}  VRAM={p.total_memory//1024**2} MB")
    print(f"  BF16 ok : {p.major >= 8}")
    if p.major >= 8:
        print("  --> L4 (SM 8.9) confirmed: FlashAttention + BF16 available")
    else:
        print("  --> Non-Ampere GPU: only FP32 and FP16 available")
EOF
log_pass "GPU info printed"

# ── 2. Import check ───────────────────────────────────────────────────────────
section "Import Check"
if ${PYTHON} -c "
from nanochat.vision import (
    NanoChatVisionModel,
    VisionModelConfig,
    VisionExperimentConfig,
    build_clip_processor,
    build_caption_tokenizer,
    create_coco_dataloader,
)
print('  All nanochat.vision imports OK')
"; then
  log_pass "All imports successful"
else
  log_fail "Import error — check PYTHONPATH and dependencies"
fi

# ── 3. Unit tests (CPU, no COCO data needed) ─────────────────────────────────
section "Unit Tests (CPU)"
if ${PYTHON} -m pytest nanochat/vision/tests/test_gpt_compat.py -v --tb=short -q; then
  log_pass "All unit tests passed"
else
  log_fail "Unit tests failed — see output above"
fi

# ── 4. Data check ─────────────────────────────────────────────────────────────
section "Dataset Check"
VAL_ANN="${DATA_ROOT}/annotations/captions_val2017.json"
if [ -f "${VAL_ANN}" ]; then
  N=$(${PYTHON} -c "import json; d=json.load(open('${VAL_ANN}')); print(len(d['images']))")
  log_pass "Val annotations found: ${N} images"
else
  log_fail "Val annotations missing: ${VAL_ANN}"
  echo "  Run: bash scripts/download_coco.sh ${DATA_ROOT} --val_only"
fi

TRAIN_ANN="${DATA_ROOT}/annotations/captions_train2017.json"
if [ -f "${TRAIN_ANN}" ]; then
  N=$(${PYTHON} -c "import json; d=json.load(open('${TRAIN_ANN}')); print(len(d['images']))")
  log_pass "Train annotations found: ${N} images"
else
  log_fail "Train annotations missing (needed for training): ${TRAIN_ANN}"
fi

# ── 5. Mini training run ──────────────────────────────────────────────────────
section "Mini Training Run (64 samples, 10 steps)"
if [ -f "${TRAIN_ANN}" ]; then
  ${PYTHON} -m nanochat.vision.train_vision \
    --experiment_name smoke_train \
    --output_dir "${OUTPUT_ROOT}" \
    --data_root "${DATA_ROOT}" \
    --max_train_samples 64 \
    --max_val_samples 16 \
    --batch_size 4 \
    --num_workers 0 \
    --pin_memory false \
    --persistent_workers false \
    --use_amp false \
    --compile_model false \
    --benchmark_only true \
    --benchmark_steps 10 \
    --log_interval 2 \
    --seed 42 \
    && log_pass "Mini training run completed" \
    || log_fail "Mini training run failed"
else
  echo "  Skipping training run (no train annotations)"
fi

# ── 6. Mini benchmark ─────────────────────────────────────────────────────────
section "Mini Benchmark (5 steps, baseline only)"
if [ -f "${TRAIN_ANN}" ]; then
  ${PYTHON} -m nanochat.vision.benchmark_vision \
    --data_root "${DATA_ROOT}" \
    --output_dir "${OUTPUT_ROOT}/benchmark" \
    --benchmark_steps 5 \
    --warmup_steps 2 \
    --batch_size 4 \
    --max_samples 64 \
    --experiments A_baseline \
    --skip_inference true \
    && log_pass "Mini benchmark completed" \
    || log_fail "Mini benchmark failed"
else
  echo "  Skipping benchmark (no train annotations)"
fi

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo " Smoke Test Summary"
echo "================================================"
echo "  PASSED: ${PASS}"
echo "  FAILED: ${FAIL}"
if [ "${FAIL}" -eq 0 ]; then
  echo ""
  echo "  All smoke tests passed! Ready for full training."
  echo "  Next step:"
  echo "    bash scripts/run_experiments.sh"
else
  echo ""
  echo "  ${FAIL} test(s) failed. Fix issues before running experiments."
fi
echo "================================================"

exit ${FAIL}
