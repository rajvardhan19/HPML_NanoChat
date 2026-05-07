#!/usr/bin/env bash
# scripts/download_coco.sh
#
# Download and extract the COCO Captions 2017 dataset.
#
# What this downloads
# -------------------
# Annotation JSONs (~241 MB):
#   annotations/captions_train2017.json  — 118,287 image × 5 captions each
#   annotations/captions_val2017.json    —   5,000 image × 5 captions each
#
# Images (~18 GB for train, ~1 GB for val):
#   train2017/  — 118,287 JPEG images
#   val2017/    —   5,000 JPEG images
#
# GCP / L4 notes
# --------------
# On a GCP instance the download speed from images.cocodataset.org is typically
# 100–300 MB/s so the full dataset takes ~2–5 minutes.
# Use a 200 GB attached SSD (not the boot disk) to avoid space issues.
#
# Usage
# -----
#   bash scripts/download_coco.sh data/coco
#   bash scripts/download_coco.sh data/coco --val_only   # ~1.2 GB (smoke test)

set -euo pipefail

DATA_ROOT="${1:-data/coco}"
VAL_ONLY=false

# Parse optional flags
for arg in "${@:2}"; do
  case "$arg" in
    --val_only) VAL_ONLY=true ;;
  esac
done

ANNO_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
TRAIN_URL="http://images.cocodataset.org/zips/train2017.zip"
VAL_URL="http://images.cocodataset.org/zips/val2017.zip"

mkdir -p "${DATA_ROOT}"
cd "${DATA_ROOT}"

echo "=== Downloading COCO Captions 2017 to: $(pwd) ==="

# ── 1. Annotation JSON files ──────────────────────────────────────────────────
echo ""
echo "[1/3] Downloading annotations (~241 MB) ..."
if [ ! -f "annotations/captions_train2017.json" ]; then
  wget -q --show-progress -O annotations_trainval2017.zip "${ANNO_URL}"
  unzip -q annotations_trainval2017.zip
  rm -f annotations_trainval2017.zip
  echo "  ✓ Annotations extracted to annotations/"
else
  echo "  ✓ annotations/ already present, skipping"
fi

# ── 2. Validation images (~1 GB) ─────────────────────────────────────────────
echo ""
echo "[2/3] Downloading val2017 images (~1 GB) ..."
if [ ! -d "val2017" ] || [ "$(ls -1 val2017/*.jpg 2>/dev/null | wc -l)" -lt 5000 ]; then
  wget -q --show-progress -O val2017.zip "${VAL_URL}"
  unzip -q val2017.zip
  rm -f val2017.zip
  echo "  ✓ val2017/ ready ($(ls val2017/*.jpg | wc -l) images)"
else
  echo "  ✓ val2017/ already present ($(ls val2017/*.jpg | wc -l) images)"
fi

# ── 3. Training images (~18 GB) ───────────────────────────────────────────────
if [ "${VAL_ONLY}" = "false" ]; then
  echo ""
  echo "[3/3] Downloading train2017 images (~18 GB) ..."
  if [ ! -d "train2017" ] || [ "$(ls -1 train2017/*.jpg 2>/dev/null | wc -l)" -lt 118287 ]; then
    wget -q --show-progress -O train2017.zip "${TRAIN_URL}"
    unzip -q train2017.zip
    rm -f train2017.zip
    echo "  ✓ train2017/ ready ($(ls train2017/*.jpg | wc -l) images)"
  else
    echo "  ✓ train2017/ already present ($(ls train2017/*.jpg | wc -l) images)"
  fi
else
  echo "[3/3] Skipping train2017 download (--val_only)"
fi

echo ""
echo "=== COCO dataset ready at: ${DATA_ROOT} ==="
echo "  annotations/captions_train2017.json"
echo "  annotations/captions_val2017.json"
[ -d val2017   ] && echo "  val2017/   ($(ls val2017/*.jpg | wc -l) images)"
[ -d train2017 ] && echo "  train2017/ ($(ls train2017/*.jpg | wc -l) images)"
