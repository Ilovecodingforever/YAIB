#!/usr/bin/env bash
set -euo pipefail
# Canonical paired 5x5 inference workflow. No inference is launched by default.
DRY_RUN="${DRY_RUN:-1}"
MODELS="${MODELS:-LSTM}"
SOURCE_DATASETS="${SOURCE_DATASETS:-hirid hirid_15min hirid_5min miiv miiv_15min miiv_5min}"
TARGET_DATASETS="${TARGET_DATASETS:-hirid hirid_15min hirid_5min miiv miiv_15min miiv_5min}"
INCLUDE_MATCHED="${INCLUDE_MATCHED:-1}"
ONLY="${ONLY:-}" # source:model:target, for one selected configuration
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

complete_source_run() {
  local run_dir="$1" repetition fold
  for repetition in 0 1 2 3 4; do
    for fold in 0 1 2 3 4; do
      [[ -f "$run_dir/repetition_$repetition/fold_$fold/train_config.gin" ]] || return 1
      has_checkpoint "$run_dir/repetition_$repetition/fold_$fold" || return 1
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

paired_output_is_complete() {
  local output="$1" source_run="$2" source="$3" target="$4" model="$5"
  local metadata="$output/paired_eval_metadata.json"
  [[ -f "$metadata" ]] || return 1
  grep -Fq '"protocol": "paired_5x5"' "$metadata" || return 1
  grep -Fq "\"source_training_run\": \"$(readlink -f "$source_run")\"" "$metadata" || return 1
  grep -Fq "\"source_dataset\": \"$source\"" "$metadata" || return 1
  grep -Fq "\"target_dataset\": \"$target\"" "$metadata" || return 1
  grep -Fq "\"model\": \"$model\"" "$metadata" || return 1
  grep -Fq '"status": "complete"' "$metadata" || return 1
  [[ "$(find "$output" -path '*/repetition_*/fold_*/test_metrics.json' -type f | wc -l)" -eq 25 ]]
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
  if paired_output_is_complete "$output" "$source_run" "$source" "$target" "$model"; then
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
