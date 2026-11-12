"""Synthetic checks for paired checkpoint selection, splits, and resume safety."""

import json
from pathlib import Path
import subprocess

import polars as pl
import pytest

import icu_benchmarks.run as run_module
import icu_benchmarks.cross_validation as cross_validation
from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment
from icu_benchmarks.data.split_process_data import make_single_split_polars
from icu_benchmarks.paired_inference import (
    PAIRED_FOLDS,
    PAIRED_REPETITIONS,
    fold_result_is_valid,
    mark_paired_complete,
    paired_result_count,
    paired_output_status,
    prepare_paired_run,
    repair_gpu_shadowed_target_metadata,
    validate_alignment,
    validate_completed_paired_run,
    validate_source_run,
)
from icu_benchmarks.run_utils import parse_args


VARS = {"GROUP": "stay_id", "LABEL": "label"}


def source_run(tmp_path: Path, config: str = "model = 'same'\n") -> Path:
    root = tmp_path / "source-run"
    for repetition in range(PAIRED_REPETITIONS):
        for fold in range(PAIRED_FOLDS):
            fold_dir = root / f"repetition_{repetition}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            (fold_dir / "train_config.gin").write_text(config)
            (fold_dir / "model.joblib").write_text("checkpoint")
            (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": 0.5, "PR": 0.2}))
    return root


def outcomes(ids=range(20)) -> pl.DataFrame:
    ids = list(ids)
    return pl.DataFrame({"stay_id": ids, "label": [value % 2 for value in ids]})


def test_all_25_source_checkpoints_are_indexed_and_missing_is_rejected(tmp_path):
    root = source_run(tmp_path)
    source, checkpoints = validate_source_run(root / "repetition_0" / "fold_0")
    assert source == root
    assert len(checkpoints) == 25
    assert checkpoints[(4, 4)] == root / "repetition_4" / "fold_4" / "model.joblib"

    (root / "repetition_3" / "fold_2" / "model.joblib").unlink()
    with pytest.raises(ValueError, match="No usable model checkpoint"):
        validate_source_run(root / "repetition_0" / "fold_0")


def test_incompatible_source_configuration_is_rejected(tmp_path):
    root = source_run(tmp_path)
    (root / "repetition_2" / "fold_1" / "train_config.gin").write_text("model = 'different'\n")
    with pytest.raises(ValueError, match="Incompatible source fold configuration"):
        validate_source_run(root / "repetition_0" / "fold_0")


def test_source_run_requires_final_test_metrics(tmp_path):
    root = source_run(tmp_path)
    (root / "repetition_1" / "fold_3" / "test_metrics.json").unlink()
    with pytest.raises(ValueError, match="Missing or invalid source test metrics"):
        validate_source_run(root / "repetition_0" / "fold_0")


def test_corrupt_or_nonfinite_fold_metrics_are_not_valid(tmp_path):
    fold_dir = tmp_path / "fold"
    fold_dir.mkdir()
    metrics = fold_dir / "test_metrics.json"
    metrics.write_text(json.dumps({"AUC": "not-a-number", "PR": 0.2}))
    assert not fold_result_is_valid(fold_dir)
    metrics.write_text(json.dumps({"AUC": float("nan"), "PR": 0.2}))
    assert not fold_result_is_valid(fold_dir)


def test_target_outer_test_partition_is_reused_by_all_inner_folds():
    fold_splits = [
        make_single_split_polars({DataSegment.outcome: outcomes()}, VARS, 5, 0, 5, fold, seed=7)
        for fold in range(5)
    ]
    test_ids = [set(split["test"][DataSegment.outcome]["stay_id"].to_list()) for split in fold_splits]
    assert all(ids == test_ids[0] for ids in test_ids)

    repetitions = [
        set(
            make_single_split_polars({DataSegment.outcome: outcomes()}, VARS, 5, repetition, 5, 0, seed=7)[
                "test"
            ][DataSegment.outcome]["stay_id"].to_list()
        )
        for repetition in range(5)
    ]
    assert len({tuple(sorted(ids)) for ids in repetitions}) == 5


def test_alignment_rejects_different_within_hospital_labels(tmp_path):
    source_dir = tmp_path / "hirid"
    target_dir = tmp_path / "hirid_15min"
    source_dir.mkdir()
    target_dir.mkdir()
    outcomes().write_parquet(source_dir / "outc.parquet")
    outcomes().write_parquet(target_dir / "outc.parquet")
    validate_alignment(source_dir, target_dir, "hirid", "hirid_15min", 7, {"OUTCOME": "outc.parquet"}, VARS)

    changed = outcomes().with_columns(
        pl.when(pl.col("stay_id") == 3).then(0).otherwise(pl.col("label")).alias("label")
    )
    changed.write_parquet(target_dir / "outc.parquet")
    with pytest.raises(ValueError, match="stay IDs/labels differ"):
        validate_alignment(source_dir, target_dir, "hirid", "hirid_15min", 7, {"OUTCOME": "outc.parquet"}, VARS)


def test_paired_output_resume_and_completion_validation(tmp_path):
    source = source_run(tmp_path)
    checkpoints = validate_source_run(source / "repetition_0" / "fold_0")[1]
    output = tmp_path / "paired"
    metadata = prepare_paired_run(
        output,
        source,
        checkpoints,
        "hirid",
        "hirid_15min",
        "LSTM",
        7,
        tmp_path / "hirid",
        tmp_path / "hirid_15min",
    )
    assert metadata["protocol"] == "paired_5x5"
    for index in range(24):
        repetition, fold = divmod(index, 5)
        fold_dir = output / f"repetition_{repetition}" / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": 0.5, "PR": 0.2}))
    assert paired_result_count(output) == 24
    with pytest.raises(ValueError, match="24/25"):
        validate_completed_paired_run(output)

    fold_dir = output / "repetition_4" / "fold_4"
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": 0.5, "PR": 0.2}))
    validate_completed_paired_run(output)

    resumed = prepare_paired_run(
        output,
        source,
        checkpoints,
        "hirid",
        "hirid_15min",
        "LSTM",
        7,
        tmp_path / "hirid",
        tmp_path / "hirid_15min",
    )
    assert resumed["status"] == "in_progress"


def test_completed_output_requires_matching_seed_mapping_and_aggregates(tmp_path):
    source = source_run(tmp_path)
    checkpoints = validate_source_run(source / "repetition_0" / "fold_0")[1]
    output = tmp_path / "paired"
    prepare_paired_run(
        output,
        source,
        checkpoints,
        "hirid",
        "hirid_15min",
        "LSTM",
        7,
        tmp_path / "hirid",
        tmp_path / "hirid_15min",
    )
    for repetition in range(5):
        for fold in range(5):
            fold_dir = output / f"repetition_{repetition}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": 0.5, "PR": 0.2}))
    (output / "aggregated_test_metrics.json").write_text(json.dumps({"repetition_0": {}}))
    (output / "accumulated_test_metrics.json").write_text(
        json.dumps({"avg": {"AUC": 0.5, "PR": 0.2}, "std": {"AUC": 0.1, "PR": 0.1}})
    )
    mark_paired_complete(output)
    assert (
        paired_output_status(
            output,
            source,
            checkpoints,
            "hirid",
            "hirid_15min",
            "LSTM",
            7,
            tmp_path / "hirid",
            tmp_path / "hirid_15min",
        )
        == "complete"
    )
    with pytest.raises(ValueError, match="does not match"):
        paired_output_status(
            output,
            source,
            checkpoints,
            "hirid",
            "hirid_15min",
            "LSTM",
            8,
            tmp_path / "hirid",
            tmp_path / "hirid_15min",
        )
    mismatched = dict(checkpoints)
    mismatched[(0, 0)] = tmp_path / "different-checkpoint"
    with pytest.raises(ValueError, match="does not match"):
        paired_output_status(
            output,
            source,
            mismatched,
            "hirid",
            "hirid_15min",
            "LSTM",
            7,
            tmp_path / "hirid",
            tmp_path / "hirid_15min",
        )


def test_numeric_target_metadata_is_repaired_only_when_path_is_unambiguous(tmp_path):
    source = source_run(tmp_path)
    target_data = tmp_path / "data" / "hirid"
    target_data.mkdir(parents=True)
    output = tmp_path / "logs_paired_5x5" / "hirid" / "Mortality24" / "LSTM" / "_from_hirid" / "paired_5x5" / "run"
    checkpoints = validate_source_run(source / "repetition_0" / "fold_0")[1]
    metadata = prepare_paired_run(
        output,
        source,
        checkpoints,
        "hirid",
        "hirid",
        "LSTM",
        7,
        tmp_path / "data" / "hirid",
        target_data,
    )
    metadata["target_dataset"] = "0"
    (output / "paired_eval_metadata.json").write_text(json.dumps(metadata))
    sentinel = output / "repetition_0" / "fold_0"
    sentinel.mkdir(parents=True)
    metrics = sentinel / "test_metrics.json"
    metrics.write_text(json.dumps({"AUC": 0.81, "PR": 0.31}))
    assert repair_gpu_shadowed_target_metadata(output)
    assert json.loads((output / "paired_eval_metadata.json").read_text())["target_dataset"] == "hirid"
    assert json.loads(metrics.read_text())["AUC"] == 0.81


def test_run_keeps_target_name_when_cuda_is_available(tmp_path, monkeypatch):
    source_dir = tmp_path / "source" / "2026-01-01" / "repetition_0" / "fold_0"
    target_dir = tmp_path / "target"
    captured = {}

    monkeypatch.setattr(run_module, "setup_logging", lambda *args: None)
    monkeypatch.setattr(run_module, "get_config_files", lambda *args: (["BinaryClassification"], ["LSTM"]))
    monkeypatch.setattr(run_module, "get_mode", lambda: RunMode.classification)
    monkeypatch.setattr(run_module, "load_pretrained_imputation_model", lambda *args: None)
    monkeypatch.setattr(run_module, "update_wandb_config", lambda *args: None)
    monkeypatch.setattr(run_module, "log_full_line", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module.gin, "parse_config_file", lambda *args: None)
    monkeypatch.setattr(run_module.gin, "query_parameter", lambda key: {"GROUP": "stay_id", "LABEL": "label"})
    monkeypatch.setattr(run_module, "resolve_gin_value", lambda value: value)
    monkeypatch.setattr(run_module, "validate_source_run", lambda path: (source_dir.parent.parent, {}))
    monkeypatch.setattr(run_module, "validate_alignment", lambda *args: captured.update(source=args[2], target=args[3]))

    def fake_prepare(run_dir, source_run_path, checkpoints, source_name, target_name, *args):
        captured.update(output=run_dir, metadata_target=target_name)
        return {}

    monkeypatch.setattr(run_module, "prepare_paired_run", fake_prepare)
    monkeypatch.setattr(run_module, "execute_repeated_cv", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "validate_completed_paired_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "aggregate_results", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_module, "mark_paired_complete", lambda *args: None)
    monkeypatch.setattr(run_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(run_module.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(run_module.torch.cuda, "get_device_name", lambda index: "fake-gpu")

    run_module.main(
        [
            "-d",
            str(target_dir),
            "-t",
            "BinaryClassification",
            "-n",
            "hirid",
            "-tn",
            "Mortality24",
            "-m",
            "LSTM",
            "--eval",
            "--paired-fold-eval",
            "-sn",
            "hirid",
            "--source-dir",
            str(source_dir),
            "--paired-source-data-dir",
            str(target_dir),
            "--paired-source-seed",
            "7",
            "-s",
            "7",
            "--paired-run-dir",
            str(tmp_path / "output"),
        ]
    )
    assert captured == {
        "source": "hirid",
        "target": "hirid",
        "output": tmp_path / "output",
        "metadata_target": "hirid",
    }


def test_paired_cv_uses_matching_source_checkpoint_and_skips_completed(tmp_path, monkeypatch):
    source = source_run(tmp_path)
    output = tmp_path / "output"
    calls = []
    completed = output / "repetition_0" / "fold_0"
    completed.mkdir(parents=True)
    (completed / "test_metrics.json").write_text(json.dumps({"AUC": 0.5, "PR": 0.2}))

    def fake_preprocess(*args, **kwargs):
        calls.append((kwargs["repetition_index"], kwargs["fold_index"]))
        return {}, None

    def fake_train(data, log_dir, source_dir, **kwargs):
        (log_dir / "test_metrics.json").write_text(json.dumps({"AUC": 0.5, "PR": 0.2}))
        calls.append(("source", Path(source_dir).relative_to(source)))
        return 0

    monkeypatch.setattr(cross_validation, "preprocess_data", fake_preprocess)
    monkeypatch.setattr(cross_validation, "train_common", fake_train)
    cross_validation.execute_repeated_cv(
        tmp_path / "target",
        output,
        seed=7,
        eval_only=True,
        load_weights=True,
        source_dir=source / "repetition_0" / "fold_0",
        mode=RunMode.classification,
        paired_fold_eval=True,
    )
    source_calls = [value for value in calls if value[0] == "source"]
    assert len(source_calls) == 24
    assert source_calls[0][1] == Path("repetition_0/fold_1")
    assert source_calls[-1][1] == Path("repetition_4/fold_4")


def test_cli_rejects_unpaired_or_unsafe_modes():
    common = [
        "-d",
        "target",
        "-t",
        "BinaryClassification",
        "-n",
        "hirid_15min",
        "-sn",
        "hirid",
        "--source-dir",
        "source/repetition_0/fold_0",
        "--paired-source-data-dir",
        "source-data",
        "--paired-source-seed",
        "7",
        "-s",
        "7",
    ]
    assert parse_args([*common, "--eval", "--paired-fold-eval"]).paired_fold_eval
    with pytest.raises(SystemExit):
        parse_args([*common, "--paired-fold-eval"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--eval", "--paired-fold-eval", "--fine-tune", "10"])
    with pytest.raises(SystemExit):
        parse_args([*common, "--eval", "--paired-fold-eval", "--paired-source-seed", "8"])


def test_script_dry_run_discovers_one_configuration(tmp_path):
    logs = tmp_path / "logs"
    run = logs / "hirid" / "Mortality24" / "LSTM" / "2026-01-01T00-00-00"
    for repetition in range(5):
        for fold in range(5):
            fold_dir = run / f"repetition_{repetition}" / f"fold_{fold}"
            fold_dir.mkdir(parents=True)
            (fold_dir / "train_config.gin").write_text("config")
            (fold_dir / "model.joblib").write_text("checkpoint")
            (fold_dir / "test_metrics.json").write_text(json.dumps({"AUC": 0.5, "PR": 0.2}))
    (tmp_path / "data" / "hirid").mkdir(parents=True)
    (tmp_path / "data" / "hirid_15min").mkdir(parents=True)
    result = subprocess.run(
        ["bash", "run_remaining_yaib_inference_conda.sh"],
        cwd=Path(__file__).parent.parent,
        env={
            "DRY_RUN": "1",
            "MODELS": "LSTM",
            "SOURCE_DATASETS": "hirid",
            "TARGET_DATASETS": "hirid_15min",
            "ONLY": "hirid:LSTM:hirid_15min",
            "LOGS": str(logs),
            "DATA_ROOT": str(tmp_path / "data"),
            "PAIRED_LOGS": str(tmp_path / "paired"),
            "PATH": "/usr/bin:/bin",
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--paired-fold-eval" in result.stdout
