#!/usr/bin/env bash
set -euo pipefail
# Existing-checkpoint evaluation only. Run with DRY_RUN=0 to execute.
# By default, generates only missing LSTM configurations; set MODELS="LSTM GRUD Transformer" for all.
DRY_RUN="${DRY_RUN:-1}"
YAIB_ENV="/home/mingzhul/projects/Simultaneous-EHR/repos/YAIB/.conda/yaib"
YAIB_CLI="$YAIB_ENV/bin/icu-benchmarks"
export PATH="$YAIB_ENV/bin:$PATH"
if [[ "$DRY_RUN" != "1" && ! -x "$YAIB_CLI" ]]; then
  echo "ERROR: icu-benchmarks not found or not executable: $YAIB_CLI" >&2
  echo "Activate the YAIB environment and verify the installation." >&2
  exit 1
fi
MODELS=" ${MODELS:-LSTM} "
ROOT="/heinz-georgenas/users/mingzhul/Simultaneous-EHR/yaib"
LOGS="$ROOT/logs"
TASK="Mortality24"
run_eval() {
  local model="$1" source="$2" test="$3" run="$4"
  local checkpoint="$LOGS/$run/repetition_0/fold_0"
  local datadir="$ROOT/data/mortality24/$test"
  if [[ "$MODELS" != *" $model "* ]]; then return; fi
  if [[ ! -f "$checkpoint/train_config.gin" ]]; then
    echo "SKIP (missing checkpoint config): $checkpoint" >&2; return
  fi
  if [[ ! -d "$datadir" ]]; then
    echo "SKIP (missing test data): $datadir" >&2; return
  fi
  local cmd=("$YAIB_CLI" --eval -d "$datadir" -t BinaryClassification -tn "$TASK" -m "$model" -s 0 -l "$LOGS" -sn "$source" -n "$test" --source-dir "$checkpoint")
  if [[ "$DRY_RUN" == "1" ]]; then printf "%q " "${cmd[@]}"; echo; else "${cmd[@]}"; fi
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
