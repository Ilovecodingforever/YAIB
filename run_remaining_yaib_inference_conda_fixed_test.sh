#!/usr/bin/env bash
set -euo pipefail

# Evaluate existing YAIB checkpoints on one fixed target outer test partition.
#
# WHY:
# YAIB normally runs --eval over 5 repetitions x 5 inner folds (25 scores).
# The outer test IDs depend on the repetition, NOT the inner fold: there are
# only 5 distinct outer test partitions, each repeated for 5 inner folds.
# A checkpoint trained at repetition_0/fold_0 has already seen the development
# patients from outer repetitions 1-4. Therefore, evaluating it on those
# repetitions at another frequency in the SAME hospital causes patient leakage.
# Restricting inference to repetition_0/fold_0 uses only the source checkpoint's
# held-out outer test patients (assuming identical stay IDs/labels and split seed
# across the three frequencies). This also avoids 5x redundant testing of the
# same outer test partition.
#
# DO NOT change cv_repetitions or cv_folds from 5 to 1: these parameters define
# the patient partition itself. We only set *_to_train=1, i.e. run the first
# existing outer repetition and its first inner fold.
#
# Source checkpoints and repository files stay untouched. YAIB's --eval branch
# loads its Gin configuration from --source-dir/train_config.gin and does not
# apply ordinary -hp overrides in this version, so the script creates a
# temporary COPY of the config and symlinks the existing checkpoint weights.
#
# NOTE: The evaluation pipeline still fits its preprocessing on target data.
# This change fixes test-set membership, not source-vs-target scaling behavior.
# Existing 25-evaluation result summaries should not be compared directly to
# these one-test-fold results. Re-evaluate matched-frequency controls with the
# same checkpoint/seed/protocol for fair frequency comparisons.
#
# Prerequisites: same eligible stays, labels, and split seed across the 1h,
# 15min, and 5min train/test datasets within each hospital; otherwise IDs may differ.
# The split is at YAIB's stay-ID level; check subject-level disjointness if
# individuals can have multiple stays.
#
# DRY_RUN=1 (default) prints the evaluation commands without writing anything.
# DRY_RUN=0 executes. MODELS defaults to LSTM (as in the original script).
# INCLUDE_MATCHED=1 (default) also runs matched-frequency inference controls.
# Source config copies are placed in TMPDIR and removed after each run.

DRY_RUN="${DRY_RUN:-1}"
MODELS=" ${MODELS:-LSTM} "
INCLUDE_MATCHED="${INCLUDE_MATCHED:-1}"
SEED="${SEED:-0}"
YAIB_ENV="/home/mingzhul/projects/Simultaneous-EHR/repos/YAIB/.conda/yaib"
YAIB_CLI="$YAIB_ENV/bin/icu-benchmarks"
export PATH="$YAIB_ENV/bin:$PATH"

ROOT="/heinz-georgenas/users/mingzhul/Simultaneous-EHR/yaib"
LOGS="$ROOT/logs"
TASK="Mortality24"

if [[ "$DRY_RUN" != "1" && ! -x "$YAIB_CLI" ]]; then
  echo "ERROR: executable not found: $YAIB_CLI" >&2
  exit 1
fi

run_eval() {
  local model="$1" source="$2" test="$3" run="$4"
  local checkpoint="$LOGS/$run/repetition_0/fold_0"
  local datadir="$ROOT/data/mortality24/$test"

  [[ "$MODELS" == *" $model "* ]] || return 0

  # A temporary config is required because --eval loads train_config.gin from
  # --source-dir; writing to the real checkpoint would alter saved experiments.
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY RUN: stage checkpoint config/weights from $checkpoint with:"
    echo "  execute_repeated_cv.cv_repetitions_to_train = 1"
    echo "  execute_repeated_cv.cv_folds_to_train = 1"
    printf '%q ' "$YAIB_CLI" --eval -d "$datadir" -t BinaryClassification \
      -tn "$TASK" -m "$model" -s "$SEED" -l "$LOGS" -sn "$source" \
      -n "$test" --source-dir "<temporary-config-for-$source-$model>"
    echo
    return 0
  fi

  if [[ ! -f "$checkpoint/train_config.gin" ]]; then
    echo "SKIP (missing source config): $checkpoint" >&2
    return 0
  fi
  if [[ ! -d "$datadir" ]]; then
    echo "SKIP (missing target data directory): $datadir" >&2
    return 0
  fi
  # Reject unfinished tuning runs and directories without trained weights.
  local weight="" candidate
  for candidate in model.ckpt model-v1.ckpt last.ckpt model.joblib last.joblib; do
    if [[ -f "$checkpoint/$candidate" ]]; then
      weight="$candidate"
      break
    fi
  done
  if [[ -z "$weight" ]]; then
    echo "SKIP (no trained checkpoint): $checkpoint" >&2
    return 0
  fi

  local eval_source
  eval_source="$(mktemp -d "${TMPDIR:-/tmp}/yaib-eval-fold0.XXXXXXXX")"
  # Trap scoped to this function's subshell so temp files are removed on
  # success, failure, or interruption without touching original checkpoints.
  (
    trap 'rm -rf -- "$eval_source"' EXIT
    cp -- "$checkpoint/train_config.gin" "$eval_source/train_config.gin"

    # Only append runtime execution limits; original 5x5 split remains.
    cat >> "$eval_source/train_config.gin" <<'GIN'

# Inference-only: test source checkpoint on its held-out outer split.
execute_repeated_cv.cv_repetitions_to_train = 1
execute_repeated_cv.cv_folds_to_train = 1
GIN

    # Link the existing model weights. No retraining and no duplicate weights.
    for candidate in model.ckpt model-v1.ckpt last.ckpt model.joblib last.joblib; do
      if [[ -f "$checkpoint/$candidate" ]]; then
        ln -s -- "$checkpoint/$candidate" "$eval_source/$candidate"
      fi
    done
    echo "EVAL: $model | $source -> $test | seed=$SEED | target repetition_0/fold_0"
    "$YAIB_CLI" --eval -d "$datadir" -t BinaryClassification \
      -tn "$TASK" -m "$model" -s "$SEED" -l "$LOGS" \
      -sn "$source" -n "$test" --source-dir "$eval_source"
  )
}

run_eval GRUD hirid hirid_15min "hirid/Mortality24/GRUD/2026-10-29T13-30-50"
run_eval GRUD hirid hirid_5min "hirid/Mortality24/GRUD/2026-10-29T13-30-50"
run_eval GRUD hirid miiv_15min "hirid/Mortality24/GRUD/2026-10-29T13-30-50"
run_eval GRUD hirid miiv_5min "hirid/Mortality24/GRUD/2026-10-29T13-30-50"
run_eval LSTM hirid hirid_15min "hirid/Mortality24/LSTM/2026-10-26T16-38-10"
run_eval LSTM hirid hirid_5min "hirid/Mortality24/LSTM/2026-10-26T16-38-10"
run_eval LSTM hirid miiv_15min "hirid/Mortality24/LSTM/2026-10-26T16-38-10"
run_eval LSTM hirid miiv_5min "hirid/Mortality24/LSTM/2026-10-26T16-38-10"
run_eval Transformer hirid hirid_15min "hirid/Mortality24/Transformer/2026-11-02T15-58-10"
run_eval Transformer hirid hirid_5min "hirid/Mortality24/Transformer/2026-11-02T15-58-10"
run_eval Transformer hirid miiv "hirid/Mortality24/Transformer/2026-11-02T15-58-10"
run_eval Transformer hirid miiv_15min "hirid/Mortality24/Transformer/2026-11-02T15-58-10"
run_eval Transformer hirid miiv_5min "hirid/Mortality24/Transformer/2026-11-02T15-58-10"
run_eval GRUD hirid_15min hirid_5min "hirid_15min/Mortality24/GRUD/2026-10-31T11-17-17"
run_eval GRUD hirid_15min miiv_15min "hirid_15min/Mortality24/GRUD/2026-10-31T11-17-17"
run_eval GRUD hirid_15min miiv_5min "hirid_15min/Mortality24/GRUD/2026-10-31T11-17-17"
run_eval LSTM hirid_15min hirid_5min "hirid_15min/Mortality24/LSTM/2026-10-27T11-43-55"
run_eval LSTM hirid_15min miiv_15min "hirid_15min/Mortality24/LSTM/2026-10-27T11-43-55"
run_eval LSTM hirid_15min miiv_5min "hirid_15min/Mortality24/LSTM/2026-10-27T11-43-55"
run_eval Transformer hirid_15min hirid_5min "hirid_15min/Mortality24/Transformer/2026-10-28T12-33-17"
run_eval Transformer hirid_15min miiv "hirid_15min/Mortality24/Transformer/2026-10-28T12-33-17"
run_eval Transformer hirid_15min miiv_15min "hirid_15min/Mortality24/Transformer/2026-10-28T12-33-17"
run_eval Transformer hirid_15min miiv_5min "hirid_15min/Mortality24/Transformer/2026-10-28T12-33-17"
run_eval GRUD hirid_5min hirid "hirid_5min/Mortality24/GRUD/2026-10-28T19-21-50"
run_eval GRUD hirid_5min hirid_15min "hirid_5min/Mortality24/GRUD/2026-10-28T19-21-50"
run_eval GRUD hirid_5min miiv "hirid_5min/Mortality24/GRUD/2026-10-28T19-21-50"
run_eval GRUD hirid_5min miiv_15min "hirid_5min/Mortality24/GRUD/2026-10-28T19-21-50"
run_eval GRUD hirid_5min miiv_5min "hirid_5min/Mortality24/GRUD/2026-10-28T19-21-50"
run_eval LSTM hirid_5min hirid_15min "hirid_5min/Mortality24/LSTM/2026-10-27T11-39-46"
run_eval LSTM hirid_5min miiv_15min "hirid_5min/Mortality24/LSTM/2026-10-27T11-39-46"
run_eval LSTM hirid_5min miiv_5min "hirid_5min/Mortality24/LSTM/2026-10-27T11-39-46"
run_eval Transformer hirid_5min hirid_15min "hirid_5min/Mortality24/Transformer/2026-11-03T12-41-33"
run_eval Transformer hirid_5min miiv "hirid_5min/Mortality24/Transformer/2026-11-03T12-41-33"
run_eval Transformer hirid_5min miiv_15min "hirid_5min/Mortality24/Transformer/2026-11-03T12-41-33"
run_eval Transformer hirid_5min miiv_5min "hirid_5min/Mortality24/Transformer/2026-11-03T12-41-33"
run_eval LSTM miiv hirid "miiv/Mortality24/LSTM/2026-10-26T16-39-18"
run_eval LSTM miiv hirid_15min "miiv/Mortality24/LSTM/2026-10-26T16-39-18"
run_eval LSTM miiv hirid_5min "miiv/Mortality24/LSTM/2026-10-26T16-39-18"
run_eval LSTM miiv miiv_15min "miiv/Mortality24/LSTM/2026-10-26T16-39-18"
run_eval LSTM miiv miiv_5min "miiv/Mortality24/LSTM/2026-10-26T16-39-18"
run_eval Transformer miiv hirid "miiv/Mortality24/Transformer/2026-11-02T00-27-19"
run_eval Transformer miiv hirid_15min "miiv/Mortality24/Transformer/2026-11-02T00-27-19"
run_eval Transformer miiv hirid_5min "miiv/Mortality24/Transformer/2026-11-02T00-27-19"
run_eval Transformer miiv miiv_15min "miiv/Mortality24/Transformer/2026-11-02T00-27-19"
run_eval Transformer miiv miiv_5min "miiv/Mortality24/Transformer/2026-11-02T00-27-19"
run_eval LSTM miiv_15min hirid "miiv_15min/Mortality24/LSTM/2026-11-04T00-49-11"
run_eval LSTM miiv_15min hirid_15min "miiv_15min/Mortality24/LSTM/2026-11-04T00-49-11"
run_eval LSTM miiv_15min hirid_5min "miiv_15min/Mortality24/LSTM/2026-11-04T00-49-11"
run_eval LSTM miiv_15min miiv "miiv_15min/Mortality24/LSTM/2026-11-04T00-49-11"
run_eval LSTM miiv_15min miiv_5min "miiv_15min/Mortality24/LSTM/2026-11-04T00-49-11"
run_eval LSTM miiv_5min hirid "miiv_5min/Mortality24/LSTM/2026-11-05T00-24-44"
run_eval LSTM miiv_5min hirid_15min "miiv_5min/Mortality24/LSTM/2026-11-05T00-24-44"
run_eval LSTM miiv_5min hirid_5min "miiv_5min/Mortality24/LSTM/2026-11-05T00-24-44"
run_eval LSTM miiv_5min miiv "miiv_5min/Mortality24/LSTM/2026-11-05T00-24-44"
run_eval LSTM miiv_5min miiv_15min "miiv_5min/Mortality24/LSTM/2026-11-05T00-24-44"

run_eval GRUD hirid miiv "hirid/Mortality24/GRUD/2026-10-29T13-30-50"
run_eval LSTM hirid miiv "hirid/Mortality24/LSTM/2026-10-26T16-38-10"
run_eval GRUD hirid_15min hirid "hirid_15min/Mortality24/GRUD/2026-10-31T11-17-17"
run_eval GRUD hirid_15min miiv "hirid_15min/Mortality24/GRUD/2026-10-31T11-17-17"
run_eval LSTM hirid_15min hirid "hirid_15min/Mortality24/LSTM/2026-10-27T11-43-55"
run_eval LSTM hirid_15min miiv "hirid_15min/Mortality24/LSTM/2026-10-27T11-43-55"
run_eval Transformer hirid_15min hirid "hirid_15min/Mortality24/Transformer/2026-10-28T12-33-17"
run_eval LSTM hirid_5min hirid "hirid_5min/Mortality24/LSTM/2026-10-27T11-39-46"
run_eval LSTM hirid_5min miiv "hirid_5min/Mortality24/LSTM/2026-10-27T11-39-46"
run_eval Transformer hirid_5min hirid "hirid_5min/Mortality24/Transformer/2026-11-03T12-41-33"
run_eval GRUD miiv hirid "miiv/Mortality24/GRUD/2026-11-03T14-14-31"
run_eval GRUD miiv hirid_15min "miiv/Mortality24/GRUD/2026-11-03T14-14-31"
run_eval GRUD miiv hirid_5min "miiv/Mortality24/GRUD/2026-11-03T14-14-31"
run_eval GRUD miiv miiv_15min "miiv/Mortality24/GRUD/2026-11-03T14-14-31"
run_eval GRUD miiv miiv_5min "miiv/Mortality24/GRUD/2026-11-03T14-14-31"

# Matched-frequency inference-only controls, using the SAME source checkpoint
# and target repetition_0/fold_0 as the mismatched evaluations. The original
# training-CV aggregated baseline is NOT directly comparable: it combines
# different trained source folds whereas --eval uses one selected checkpoint.
if [[ "$INCLUDE_MATCHED" == "1" ]]; then
  run_eval GRUD hirid hirid "hirid/Mortality24/GRUD/2026-10-29T13-30-50"
  run_eval LSTM hirid hirid "hirid/Mortality24/LSTM/2026-10-26T16-38-10"
  run_eval Transformer hirid hirid "hirid/Mortality24/Transformer/2026-11-02T15-58-10"
  run_eval GRUD hirid_15min hirid_15min "hirid_15min/Mortality24/GRUD/2026-10-31T11-17-17"
  run_eval LSTM hirid_15min hirid_15min "hirid_15min/Mortality24/LSTM/2026-10-27T11-43-55"
  run_eval Transformer hirid_15min hirid_15min "hirid_15min/Mortality24/Transformer/2026-10-28T12-33-17"
  run_eval GRUD hirid_5min hirid_5min "hirid_5min/Mortality24/GRUD/2026-10-28T19-21-50"
  run_eval LSTM hirid_5min hirid_5min "hirid_5min/Mortality24/LSTM/2026-10-27T11-39-46"
  run_eval Transformer hirid_5min hirid_5min "hirid_5min/Mortality24/Transformer/2026-11-03T12-41-33"
  run_eval LSTM miiv miiv "miiv/Mortality24/LSTM/2026-10-26T16-39-18"
  run_eval Transformer miiv miiv "miiv/Mortality24/Transformer/2026-11-02T00-27-19"
  run_eval LSTM miiv_15min miiv_15min "miiv_15min/Mortality24/LSTM/2026-11-04T00-49-11"
  run_eval LSTM miiv_5min miiv_5min "miiv_5min/Mortality24/LSTM/2026-11-05T00-24-44"
  run_eval GRUD miiv miiv "miiv/Mortality24/GRUD/2026-11-03T14-14-31"
fi
