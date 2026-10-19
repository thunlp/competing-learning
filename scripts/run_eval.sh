#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# -eq 1 && "$1" == --help ]]; then
  "${PYTHON:-python}" -m evaluation.run --help
  exit 0
fi
if [[ $# -lt 2 ]]; then
  echo 'Usage: bash scripts/run_eval.sh MODEL_DIR OUTPUT_DIR [additional evaluation arguments]' >&2
  echo 'Prepare data first: python -m evaluation.prepare_data --benchmark lm' >&2
  exit 2
fi
MODEL=$1
OUTPUT=$2
shift 2
exec "${PYTHON:-python}" -m evaluation.run --model "$MODEL" --output "$OUTPUT" "$@"
