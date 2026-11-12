#!/usr/bin/env bash
set -euo pipefail
# Canonical paired 5x5 inference workflow. No inference is launched by default.
DRY_RUN="${DRY_RUN:-1}"
MODELS="${MODELS:-LSTM}"
SOURCE_DATASETS="${SOURCE_DATASETS:-hirid hirid_15min hirid_5min miiv miiv_15min miiv_5min}"
TARGET_DATASETS="${TARGET_DATASETS:-hirid hirid_15min hirid_5min miiv miiv_15min miiv_5min}"
INCLUDE_MATCHED="${INCLUDE_MATCHED:-1}"
ONLY="${ONLY:-}" # source:model:target, for one selected configuration
# One explicit source-seed assertion applies to every selected source run.
# Run separate subsets when source runs were created with different seeds.
SEED="${SEED:-}"
YAIB_ENV="${YAIB_ENV:-/home/mingzhul/projects/Simultaneous-EHR/repos/YAIB/.conda/yaib}"
YAIB_CLI="$YAIB_ENV/bin/icu-benchmarks"
ROOT="${ROOT:-/heinz-georgenas/users/mingzhul/Simultaneous-EHR/yaib}"
LOGS="${LOGS:-$ROOT/logs}"
DATA_ROOT="${DATA_ROOT:-$ROOT/data/mortality24}"
PAIRED_LOGS="${PAIRED_LOGS:-$ROOT/logs_paired_5x5}"
TASK="${TASK:-Mortality24}"

if [[ "$DRY_RUN" != "1" && ! -x "$YAIB_CLI" ]]; then
  echo "ERROR: icu-benchmarks not found or not executable: $YAIB_CLI" >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && -z "$SEED" ]]; then
  echo "ERROR: set SEED to the verified seed used by the selected source training run." >&2
  exit 1
fi
contains() {
  local haystack=" $1 " needle=" $2 "
  [[ "$haystack" == *"$needle"* ]]
}

has_checkpoint() {
  local fold_dir="$1" candidate
  for candidate in model.ckpt model-v1.ckpt last.ckpt model.joblib last.joblib; do
    [[ -f "$fold_dir/$candidate" ]] && return 0
  done
  return 1
}

valid_test_metrics() {
  python3 -c 'import json, math, sys; d=json.load(open(sys.argv[1])); ok=isinstance(d, dict) and all(isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool) and math.isfinite(float(d[k])) for k in ("AUC", "PR")); raise SystemExit(0 if ok else 1)' "$1"
}

complete_source_run() {
  local run_dir="$1" repetition fold
  for repetition in 0 1 2 3 4; do
    for fold in 0 1 2 3 4; do
      [[ -f "$run_dir/repetition_$repetition/fold_$fold/train_config.gin" ]] || return 1
      local fold_dir="$run_dir/repetition_$repetition/fold_$fold"
      has_checkpoint "$fold_dir" || return 1
      [[ -f "$fold_dir/test_metrics.json" ]] && valid_test_metrics "$fold_dir/test_metrics.json" || return 1
    done
  done
}

latest_source_run() {
  local source="$1" model="$2" base="$LOGS/$source/$TASK/$model"
  local run_dir run_name selected="" saw_selected=0
  [[ -d "$base" ]] || return 0
  while IFS= read -r run_dir; do
    run_name="$(basename "$run_dir")"
    [[ "$run_name" == tmp* ]] && continue
    [[ "$run_name" == _from_* ]] && continue
    [[ "$run_name" == 20* ]] || continue
    if complete_source_run "$run_dir"; then
      selected="$run_dir"
      saw_selected=1
    elif [[ "$saw_selected" == "1" ]]; then
      echo "NOTICE: newer incomplete source run ignored: $run_dir" >&2
    else
      echo "NOTICE: incomplete source run ignored: $run_dir" >&2
    fi
  done < <(find "$base" -mindepth 1 -maxdepth 1 -type d -print | sort)
  if [[ -n "$selected" ]]; then
    printf '%s\n' "$selected"
  fi
  return 0
}

paired_output_status() {
  local output="$1" source_run="$2" source="$3" target="$4" model="$5" seed="$6"
  "$YAIB_ENV/bin/python" -c '
import sys
from pathlib import Path
from icu_benchmarks.paired_inference import paired_output_status, validate_source_run
output, source_run, source, target, model, seed, source_data, target_data = sys.argv[1:]
_, checkpoints = validate_source_run(Path(source_run) / "repetition_0" / "fold_0")
print(paired_output_status(Path(output), Path(source_run), checkpoints, source, target, model, int(seed), Path(source_data), Path(target_data)))
' "$output" "$source_run" "$source" "$target" "$model" "$seed" "$DATA_ROOT/$source" "$DATA_ROOT/$target"
}

run_eval() {
  local model="$1" source="$2" target="$3" source_run="$4"
  local source_data="$DATA_ROOT/$source" target_data="$DATA_ROOT/$target"
  local checkpoint="$source_run/repetition_0/fold_0"
  local output="$PAIRED_LOGS/$target/$TASK/$model/_from_$source/paired_5x5/$(basename "$source_run")"
  local seed_arg="${SEED:-<verified-source-seed>}"
  local cmd

  contains "$MODELS" "$model" || return 0
  contains "$SOURCE_DATASETS" "$source" || return 0
  contains "$TARGET_DATASETS" "$target" || return 0
  [[ "$INCLUDE_MATCHED" == "1" || "$source" != "$target" ]] || return 0
  [[ -z "$ONLY" || "$ONLY" == "$source:$model:$target" ]] || return 0

  if [[ ! -d "$source_data" || ! -d "$target_data" ]]; then
    echo "SKIP: missing processed data for $source -> $target" >&2
    return 0
  fi
  local output_status="new"
  if [[ -n "$SEED" ]]; then
    if ! output_status="$(paired_output_status "$output" "$source_run" "$source" "$target" "$model" "$SEED")"; then
      echo "ERROR: paired output metadata is incompatible or unreadable: $output" >&2
      return 1
    fi
  fi
  if [[ "$output_status" == "complete" ]]; then
    echo "SKIP: completed paired run $source | $model | $target"
    return 0
  fi

  cmd=("$YAIB_CLI" --eval --paired-fold-eval
    -d "$target_data" -t BinaryClassification -tn "$TASK" -m "$model"
    -s "$seed_arg" --paired-source-seed "$seed_arg"
    -l "$PAIRED_LOGS" -sn "$source" -n "$target"
    --source-dir "$checkpoint" --paired-source-data-dir "$source_data"
    --paired-run-dir "$output")

  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY RUN: '
    printf '%q ' "${cmd[@]}"
    printf '\n'
  else
    echo "EVAL: $source | $model | $target | source_run=$(basename "$source_run") | seed=$SEED"
    "${cmd[@]}"
  fi
}
planned=0
failed=0
for model in $MODELS; do
  for source in $SOURCE_DATASETS; do
    source_run="$(latest_source_run "$source" "$model")"
    if [[ -z "$source_run" ]]; then
      echo "SKIP: no completed 25-checkpoint source run for $source/$model" >&2
      continue
    fi
    for target in $TARGET_DATASETS; do
      if [[ -n "$ONLY" && "$ONLY" != "$source:$model:$target" ]]; then
        continue
      fi
      if run_eval "$model" "$source" "$target" "$source_run"; then
        planned=$((planned + 1))
      else
        echo "FAILED: $source | $model | $target" >&2
        failed=$((failed + 1))
      fi
    done
  done
done

echo "SUMMARY: configurations=$planned failures=$failed dry_run=$DRY_RUN"
[[ "$failed" -eq 0 ]]
