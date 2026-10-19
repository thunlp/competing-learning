#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
MODEL=${1:-390m}
METHOD=${2:-both}
case "$MODEL" in
  180m|250m) GPUS=1 ;;
  390m|510m|750m|gdn-400m|gla-400m) GPUS=2 ;;
  1b) GPUS=8 ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
EXCLUDE=g30
[[ -n "${SLURM_EXCLUDE:-}" ]] && EXCLUDE+=",${SLURM_EXCLUDE}"
[[ -n "${SBATCH_EXCLUDE:-}" ]] && EXCLUDE+=",${SBATCH_EXCLUDE}"
PARTITION_ARGS=()
[[ -n "${PARTITION:-}" ]] && PARTITION_ARGS=(--partition="$PARTITION")
mkdir -p "$ROOT/outputs/logs"
exec sbatch --exclude="$EXCLUDE" --nodes=1 --ntasks=1 --gres="gpu:${NPROC:-$GPUS}" \
  --cpus-per-task="${CPUS_PER_TASK:-16}" --time="${WALLTIME:-3-00:00:00}" \
  --job-name="bld-$MODEL-$METHOD" --chdir="$ROOT" \
  --output="$ROOT/outputs/logs/%x-%j.out" \
  "${PARTITION_ARGS[@]}" "$ROOT/scripts/run_main.sh" "$MODEL" "$METHOD" all
