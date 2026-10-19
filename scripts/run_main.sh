#!/usr/bin/env bash
set -euo pipefail

# Usage: DCLM_STAGE1=... DCLM_STAGE2=... bash scripts/run_main.sh [390m] [bld|vanilla|both] [all|stage1|stage2]
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
MODEL=${1:-390m}
METHOD=${2:-both}
STAGE=${3:-all}
case "$MODEL" in
  180m|250m) SCHEDULE=$MODEL; GPUS=1 ;;
  390m|510m|750m) SCHEDULE=$MODEL; GPUS=2 ;;
  1b) SCHEDULE=1b; GPUS=8 ;;
  gdn-400m|gla-400m) SCHEDULE=390m; GPUS=2 ;;
  *) echo "Unknown model: $MODEL" >&2; exit 2 ;;
esac
case "$METHOD" in bld|vanilla|both) ;; *) echo 'Method: bld, vanilla, or both' >&2; exit 2 ;; esac
case "$STAGE" in all|stage1|stage2) ;; *) echo 'Stage: all, stage1, or stage2' >&2; exit 2 ;; esac
NPROC=${NPROC:-$GPUS}
case "$NPROC" in 1|2|4|8|16) ;; *) echo 'NPROC must divide 16' >&2; exit 2 ;; esac
ACCUM=$((16 / NPROC))
TOKENIZER=${TOKENIZER:-Qwen/Qwen3-0.6B}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/outputs}
PYTHON=${PYTHON:-python}
SEED=${SEED:-0}
CONFIG_DIR=${CONFIG_DIR:-configs}
if [[ "$STAGE" != stage2 ]]; then
  : "${DCLM_STAGE1:?Set DCLM_STAGE1 to global-shard-01 parquet data}"
  [[ -d "$DCLM_STAGE1" ]] || { echo 'DCLM_STAGE1 must be a directory' >&2; exit 2; }
fi
if [[ "$STAGE" != stage1 ]]; then
  : "${DCLM_STAGE2:?Set DCLM_STAGE2 to global-shards-02-10 parquet data}"
  [[ -d "$DCLM_STAGE2" ]] || { echo 'DCLM_STAGE2 must be a directory' >&2; exit 2; }
fi
if [[ "$STAGE" == all && "$(realpath "$DCLM_STAGE1")" == "$(realpath "$DCLM_STAGE2")" ]]; then
  echo 'The two stages require disjoint DCLM data views' >&2; exit 2
fi
FAMILY=kda
[[ "$MODEL" == gdn-* ]] && FAMILY=gdn
[[ "$MODEL" == gla-* ]] && FAMILY=gla
MODEL_CONFIG="$CONFIG_DIR/model/$MODEL.json"
FREEZE=()
if [[ "$METHOD" != vanilla ]]; then
  read -r -a FREEZE <<< "$("$PYTHON" - "$MODEL_CONFIG" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    c = json.load(f)
print(*[i for i in range(c['num_hidden_layers']) if i not in c['attn']['layers']])
PY
)"
fi
launch() {
  if [[ "$NPROC" == 1 ]]; then
    "$PYTHON" train.py "$@"
  else
    "$PYTHON" -m accelerate.commands.launch --multi_gpu --num_machines=1 \
      --num_processes="$NPROC" --mixed_precision=bf16 --dynamo_backend=no \
      --main_process_port="${MASTER_PORT:-29500}" train.py "$@"
  fi
}
run_method() {
  local method=$1 checkpoint="$OUTPUT_DIR/bld/${MODEL}-${1}-seed${SEED}-stage1/model.safetensors"
  local common=(--model_name "$FAMILY" --model_config "$MODEL_CONFIG" --tok_path "$TOKENIZER"
    --output_dir "$OUTPUT_DIR" --proj_name bld --grad_accum_steps "$ACCUM" --seed "$SEED" --grad_ckpt 1)
  if [[ "$STAGE" != stage2 ]]; then
    : "${DCLM_STAGE1:?Set DCLM_STAGE1 to disjoint global-shard-01 parquet data}"
    local freeze=()
    [[ "$method" == bld ]] && freeze=(--freeze_layers "${FREEZE[@]}")
    launch "${common[@]}" --train_config "$CONFIG_DIR/training/$SCHEDULE/stage1.json" \
      --data_path "$DCLM_STAGE1" --run_name "${MODEL}-${method}-seed${SEED}-stage1" "${freeze[@]}"
  fi
  if [[ "$STAGE" != stage1 ]]; then
    : "${DCLM_STAGE2:?Set DCLM_STAGE2 to disjoint global-shards-02-10 parquet data}"
    [[ -s "$checkpoint" ]] || { echo "Missing Stage 1 weights: $checkpoint" >&2; exit 1; }
    launch "${common[@]}" --train_config "$CONFIG_DIR/training/$SCHEDULE/stage2.json" \
      --data_path "$DCLM_STAGE2" --run_name "${MODEL}-${method}-seed${SEED}-stage2" --init_from "$checkpoint"
  fi
}
if [[ "$METHOD" == both ]]; then
  run_method vanilla
  run_method bld
else
  run_method "$METHOD"
fi
