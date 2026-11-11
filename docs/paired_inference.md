# Paired 5x5 inference

`--paired-fold-eval` evaluates the source checkpoint at
`repetition_r/fold_f` on the target dataset generated with the same
`repetition_index=r` and `fold_index=f`. It requires all 25 source
checkpoints and writes one target result per pair under:

```text
logs_paired_5x5/<target>/Mortality24/<model>/_from_<source>/paired_5x5/<source-run>/
```

The run contains `repetition_0` through `repetition_4`, each with five
folds, `paired_eval_metadata.json`, `aggregated_test_metrics.json`, and
`accumulated_test_metrics.json`. The latter preserves YAIB's existing
`avg` and `std` fields. YAIB's `std` is the population standard deviation
divided by the square root of the number of results (an SEM-like value), and
the 25 values are correlated because the five inner folds in each repetition
share the same outer test partition.

Paired mode keeps YAIB's current preprocessing. In particular, preprocessing
statistics may be fitted using the target training subset during inference;
this protocol fixes checkpoint-to-test-partition pairing, not source-scaler
reuse.

The source seed is not present in the current saved Gin or Lightning metadata.
Provide the independently verified seed explicitly with both `--seed` and
`--paired-source-seed`; the CLI rejects mismatches.

The executable examples below use `SEED=0` only as a concrete example. Replace
it with the verified seed for the selected source run before launching.

## Commands

Dry-run the default LSTM source matrix:

```bash
cd /home/mingzhul/projects/Simultaneous-EHR/repos/YAIB
./run_remaining_yaib_inference_conda.sh
```

Run one selected source/target configuration:

```bash
DRY_RUN=0 SEED=0 MODELS=LSTM \
  SOURCE_DATASETS=hirid TARGET_DATASETS=hirid_15min \
  ONLY=hirid:LSTM:hirid_15min \
  ./run_remaining_yaib_inference_conda.sh
```

Run HiRID 1h LSTM to HiRID 1h LSTM using all 25 checkpoints:

```bash
DRY_RUN=0 SEED=0 MODELS=LSTM \
  SOURCE_DATASETS=hirid TARGET_DATASETS=hirid \
  ONLY=hirid:LSTM:hirid \
  ./run_remaining_yaib_inference_conda.sh
```

Resume an interrupted run by repeating the same command. The script reuses
the deterministic paired output directory, preserves valid fold results, and
evaluates only missing folds. A metadata mismatch fails instead of mixing
experiments.

Run all eligible models and source/target configurations:

```bash
DRY_RUN=0 SEED=0 MODELS="LSTM GRUD Transformer" \
  ./run_remaining_yaib_inference_conda.sh
```

