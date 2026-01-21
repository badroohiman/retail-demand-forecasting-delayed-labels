#!/usr/bin/env bash
set -euo pipefail

# ====== CONFIG (edit these) ======
IN_SAMPLE="data/processed/m5_daily_sample.parquet"
OUT_DAILY_FULL="data/processed/m5_daily_full.parquet"
OUT_LABELED_FULL="data/processed/m5_labeled_full.parquet"
OUT_FEATURES_FULL="data/processed/m5_features_full.parquet"

LABEL="y_v2"
SPLIT_DATE="2015-01-01"
SEED="42"

# If you want S3 upload, set BUCKET_PREFIX like:
#   export BUCKET_PREFIX="s3://your-bucket/m5-delayed-labels"
BUCKET_PREFIX="${BUCKET_PREFIX:-}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/ec2_pipeline_${TS}.log"
SUMMARY="reports/scale_run_summary_${TS}.md"

echo "[INFO] Logging to: ${LOG}"
echo "[INFO] Summary will be: ${SUMMARY}"

# log everything
exec > >(tee -a "$LOG") 2>&1

echo "## EC2 full-scale run summary" > "$SUMMARY"
echo "" >> "$SUMMARY"
echo "- Timestamp (UTC): ${TS}" >> "$SUMMARY"
echo "- Hostname: $(hostname)" >> "$SUMMARY"
echo "- Kernel: $(uname -r)" >> "$SUMMARY"
echo "- Python: $(python --version 2>&1)" >> "$SUMMARY"
echo "" >> "$SUMMARY"
echo "### System" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
free -h >> "$SUMMARY" || true
df -h >> "$SUMMARY" || true
echo '```' >> "$SUMMARY"
echo "" >> "$SUMMARY"

step () {
  local name="$1"
  shift
  echo ""
  echo "=============================="
  echo "[STEP] ${name}"
  echo "=============================="
  local start=$(date +%s)
  "$@"
  local end=$(date +%s)
  local dur=$((end-start))
  echo "- ${name}: ${dur}s" >> "$SUMMARY"
}

echo "### Runtime" >> "$SUMMARY"

# ---------- STEP 0: quick sanity ----------
step "Sanity check imports" bash -lc 'python - <<PY
import pandas as pd
import pyarrow as pa
import lightgbm as lgb
print("OK: pandas/pyarrow/lightgbm imported")
PY'

# ---------- STEP 1: FULL preprocess (no sampling) ----------
# If you already have OUT_DAILY_FULL, it will skip only if your preprocess script does.
# Otherwise delete it first to force regenerate.
step "Preprocess full data -> ${OUT_DAILY_FULL}" \
  python src/data/preprocess.py --out_path "${OUT_DAILY_FULL}"

# ---------- STEP 2: Label versions ----------
step "Make label versions -> ${OUT_LABELED_FULL}" \
  python src/labels/make_label_versions.py \
    --in_path "${OUT_DAILY_FULL}" \
    --out_path "${OUT_LABELED_FULL}" \
    --seed "${SEED}"

# ---------- STEP 3: Build features ----------
step "Build features -> ${OUT_FEATURES_FULL}" \
  python src/features/build_features.py \
    --in_path "${OUT_LABELED_FULL}" \
    --out_path "${OUT_FEATURES_FULL}"

# ---------- STEP 4: Train two-stage model ----------
step "Train two-stage model (label=${LABEL}, split=${SPLIT_DATE})" \
  python -m src.models.train_two_stage \
    --in_path "${OUT_FEATURES_FULL}" \
    --label "${LABEL}" \
    --split_date "${SPLIT_DATE}"

# ---------- STEP 5: capture sizes ----------
echo "" >> "$SUMMARY"
echo "### Artifact sizes" >> "$SUMMARY"
echo '```' >> "$SUMMARY"
du -sh data/processed models reports logs 2>/dev/null || true >> "$SUMMARY"
echo '```' >> "$SUMMARY"

echo ""
echo "[DONE] Pipeline completed successfully."
echo "[DONE] Summary: ${SUMMARY}"
echo "[DONE] Log: ${LOG}"

# ---------- Optional: upload to S3 ----------
if [[ -n "${BUCKET_PREFIX}" ]]; then
  echo ""
  echo "[S3] Uploading artifacts to ${BUCKET_PREFIX} ..."
  aws s3 cp "${SUMMARY}" "${BUCKET_PREFIX}/reports/" || true
  aws s3 cp "${LOG}" "${BUCKET_PREFIX}/logs/" || true
  aws s3 cp "models/" "${BUCKET_PREFIX}/models/" --recursive || true
  aws s3 cp "reports/" "${BUCKET_PREFIX}/reports/" --recursive || true
  # data upload can be large; comment out if you don't want it
  aws s3 cp "data/processed/" "${BUCKET_PREFIX}/data/processed/" --recursive || true
  echo "[S3] Upload complete."
fi
