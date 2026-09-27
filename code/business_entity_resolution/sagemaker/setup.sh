#!/usr/bin/env bash
# One-time environment setup on a SageMaker notebook instance / Studio JupyterLab (Linux, CPU only).
# Usage (from the repo root):  bash code/business_entity_resolution/sagemaker/setup.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."          # repo root (contains dataset/ and code/)

PY=""
for c in python3.13 python3.12 python3.11; do command -v "$c" >/dev/null && PY=$c && break; done
if [ -z "$PY" ]; then                  # SageMaker images ship conda: make a modern Python
  conda create -y -q -p "$PWD/.pyenv" python=3.12 >/dev/null
  PY="$PWD/.pyenv/bin/python"
fi
echo "using $($PY --version)"
$PY -m venv .venv
.venv/bin/pip install -q --upgrade pip
if ! .venv/bin/pip install -q -r code/business_entity_resolution/requirements.txt; then
  echo "pinned versions unavailable for this Python -> installing latest compatible versions"
  .venv/bin/pip install -q numpy pandas polars pyarrow scipy scikit-learn lightgbm rapidfuzz Unidecode joblib
fi
.venv/bin/python -c "import polars, lightgbm, rapidfuzz, unidecode, sklearn; print('environment OK')"
echo "CPUs: $(nproc)   RAM: $(free -g | awk '/Mem/{print $2}') GB   disk free: $(df -h . | awk 'NR==2{print $4}')"
ls dataset/train dataset/test >/dev/null && echo "dataset found" || echo "!! put the challenge files in ./dataset (train/ and test/)"
