#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA_CONFIG="${DATA_CONFIG:-${REPO_ROOT}/configs/data_config.yaml}"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/configs/model_config.yaml}"
TRAINING_CONFIG="${TRAINING_CONFIG:-${REPO_ROOT}/configs/training_config.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-${REPO_ROOT}/configs/eval_config.yaml}"

"${PYTHON}" "${REPO_ROOT}/scripts/preprocess_h5ad.py" \
  --data-config "${DATA_CONFIG}" --training-config "${TRAINING_CONFIG}"
"${PYTHON}" "${REPO_ROOT}/scripts/build_pseudobulk.py" \
  --data-config "${DATA_CONFIG}" --training-config "${TRAINING_CONFIG}"
"${PYTHON}" "${REPO_ROOT}/scripts/train.py" \
  --data-config "${DATA_CONFIG}" --model-config "${MODEL_CONFIG}" \
  --training-config "${TRAINING_CONFIG}"
"${PYTHON}" "${REPO_ROOT}/scripts/evaluate_baseline.py" \
  --data-config "${DATA_CONFIG}" --training-config "${TRAINING_CONFIG}"
"${PYTHON}" "${REPO_ROOT}/scripts/evaluate.py" \
  --data-config "${DATA_CONFIG}" --model-config "${MODEL_CONFIG}" \
  --training-config "${TRAINING_CONFIG}" --eval-config "${EVAL_CONFIG}"
