#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON=${PYTHON:-python}
OUTPUT=${OUTPUT:-outputs/retrieval}
for seed in 0 1 2; do
  "$PYTHON" -m experiments.retrieval.run --output "$OUTPUT/seed$seed/baseline" --seed "$seed"
  for step in 0 1 10 25 50 75 100 125 150 175 200 225 250 275 300 325 350 375 400 425 450 475 500 600; do
    "$PYTHON" -m experiments.retrieval.run --output "$OUTPUT/seed$seed/freeze$step" --seed "$seed" --freeze-step "$step"
    if [[ "$step" -le 500 ]]; then
      "$PYTHON" -m experiments.retrieval.run --output "$OUTPUT/seed$seed/unfreeze$step" --seed "$seed" --freeze-step "$step" --unfreeze-step 500
    fi
  done
  for percent in 0 10 20 30 40 50 60 70 80 90 100; do
    "$PYTHON" -m experiments.retrieval.run --output "$OUTPUT/seed$seed/short$percent" --seed "$seed" --short-percent "$percent"
  done
done
