#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for seed in {0..9}; do
  for step in 100 150 175; do
    "${PYTHON:-python}" -m experiments.retrieval.gradients \
      --output "${OUTPUT:-outputs/gradients}/freeze$step/seed$seed" --seed "$seed" \
      --freeze-step "$step" --unfreeze-step "$((step+50))" --steps "$((step+250))"
  done
done
