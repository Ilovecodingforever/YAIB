"""Validation and bookkeeping helpers for paired 5x5 evaluation."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import gin
import polars as pl

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment, VarType
from icu_benchmarks.data.split_process_data import make_single_split_polars


PAIRED_PROTOCOL = "paired_5x5"
PAIRED_REPETITIONS = 5
PAIRED_FOLDS = 5
WEIGHT_FILES = ("model.ckpt", "model-v1.ckpt", "last.ckpt", "model.joblib", "last.joblib")


def resolve_gin_value(value: Any) -> Any:
    """Resolve a Gin macro returned as a ConfigurableReference."""
    if not isinstance(value, gin.config.ConfigurableReference):
        return value
    if len(value.scopes) != 1:
        raise ValueError(f"Cannot resolve scoped Gin macro: {value}")
    try:
        return gin.config._CONFIG[(value.scopes[0], value.selector)]["value"]
    except KeyError as error:
        raise ValueError(f"Cannot resolve Gin macro: {value}") from error


def source_run_from_dir(source_dir: Path) -> Path:
    """Return the training-run root for a source ``repetition_0/fold_0`` path."""
    source_dir = Path(source_dir)
    if source_dir.name != "fold_0" or source_dir.parent.name != "repetition_0":
        raise ValueError(
            "Paired evaluation requires --source-dir to point to repetition_0/fold_0; "
            f"got {source_dir}"
        )
    return source_dir.parent.parent


def checkpoint_file(fold_dir: Path) -> Path:
    for filename in WEIGHT_FILES:
        candidate = fold_dir / filename
        if candidate.is_file():
            return candidate
    raise ValueError(f"No usable model checkpoint found in {fold_dir}")


def _compatible_config(config: str) -> str:
    # Source runs may carry stale fixed-test limits. Paired mode owns those
    # loop bounds, so exclude only those two bindings from compatibility checks.
    lines = []
    for line in config.splitlines():
        if "execute_repeated_cv.cv_repetitions_to_train" in line:
            continue
        if "execute_repeated_cv.cv_folds_to_train" in line:
            continue
        if line.strip() and not line.lstrip().startswith("#"):
            lines.append(" ".join(line.split()))
    return "\n".join(lines)


def validate_source_run(source_dir: Path) -> tuple[Path, dict[tuple[int, int], Path]]:
    """Validate and index all 25 source checkpoints."""
    source_dir = Path(source_dir)
    source_run = source_run_from_dir(source_dir)
    if not source_run.is_dir():
        raise ValueError(f"Source training run does not exist: {source_run}")

    checkpoints: dict[tuple[int, int], Path] = {}
    reference_config = None
    for repetition in range(PAIRED_REPETITIONS):
        for fold_index in range(PAIRED_FOLDS):
            fold_dir = source_run / f"repetition_{repetition}" / f"fold_{fold_index}"
            config_path = fold_dir / "train_config.gin"
            if not config_path.is_file():
                raise ValueError(f"Missing source configuration: {config_path}")
            checkpoint = checkpoint_file(fold_dir)
            config = _compatible_config(config_path.read_text())
            if reference_config is None:
                reference_config = config
            elif config != reference_config:
                raise ValueError(f"Incompatible source fold configuration: {config_path}")
            checkpoints[(repetition, fold_index)] = checkpoint

    return source_run, checkpoints


def _outcome_path(data_dir: Path, file_names: dict[str, str] | None) -> Path:
    filename = (file_names or {}).get(DataSegment.outcome, "outc.parquet")
    path = Path(data_dir) / filename
    if not path.is_file():
        raise ValueError(f"Missing processed outcome data: {path}")
    return path


def _outcomes(data_dir: Path, file_names: dict[str, str] | None, vars: dict[str, Any]) -> pl.DataFrame:
    group = vars[VarType.group]
    label = vars[VarType.label]
    if not isinstance(group, str) or not isinstance(label, str):
        raise ValueError("Paired alignment requires one GROUP and one LABEL column")
    outcome = pl.read_parquet(_outcome_path(data_dir, file_names))
    missing = {group, label} - set(outcome.columns)
    if missing:
        raise ValueError(f"Outcome data {data_dir} is missing columns: {sorted(missing)}")
    return outcome.group_by(group).agg(pl.col(label).max().alias(label)).sort(group)


def _split_ids(outcome: pl.DataFrame, vars: dict[str, Any], seed: int, repetition: int, fold: int) -> dict[str, set]:
    group = vars[VarType.group]
    data = make_single_split_polars(
        {DataSegment.outcome: outcome},
        vars,
        PAIRED_REPETITIONS,
        repetition,
        PAIRED_FOLDS,
        fold,
        seed=seed,
        runmode=RunMode.classification,
    )
    return {split: set(data[split][DataSegment.outcome][group].to_list()) for split in ("train", "val", "test")}


def _hospital(dataset_name: str) -> str | None:
    if dataset_name.startswith("hirid"):
        return "hirid"
    if dataset_name.startswith("miiv"):
        return "miiv"
    return None


def validate_alignment(
    source_data_dir: Path,
    target_data_dir: Path,
    source_name: str,
    target_name: str,
    seed: int,
    file_names: dict[str, str] | None,
    vars: dict[str, Any],
) -> None:
    """Validate processed outcome identity and the split protocol before inference."""
    source_outcomes = _outcomes(source_data_dir, file_names, vars)
    target_outcomes = _outcomes(target_data_dir, file_names, vars)
    source_splits = []
    target_splits = []
    for repetition in range(PAIRED_REPETITIONS):
        source_split = _split_ids(source_outcomes, vars, seed, repetition, 0)
        target_split = _split_ids(target_outcomes, vars, seed, repetition, 0)
        for split in ("train", "val", "test"):
            if source_split[split] & source_split["test"] and split != "test":
                raise ValueError(f"Source development/test overlap at repetition {repetition}: {split}")
            if target_split[split] & target_split["test"] and split != "test":
                raise ValueError(f"Target development/test overlap at repetition {repetition}: {split}")
        source_splits.append(source_split)
        target_splits.append(target_split)

    if _hospital(source_name) == _hospital(target_name):
        if source_outcomes.rows() != target_outcomes.rows():
            raise ValueError(
                "Within-hospital source and target outcome stay IDs/labels differ; "
                "paired evaluation cannot establish leakage-safe alignment"
            )
        for repetition, (source_split, target_split) in enumerate(zip(source_splits, target_splits)):
            if source_split["test"] != target_split["test"]:
                raise ValueError(f"Outer test IDs differ at repetition {repetition}")


def fold_result_is_valid(fold_dir: Path) -> bool:
    metrics_path = Path(fold_dir) / "test_metrics.json"
    if not metrics_path.is_file():
        return False
    try:
        metrics = json.loads(metrics_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(metrics, dict) and "AUC" in metrics and "PR" in metrics


def paired_result_count(run_dir: Path) -> int:
    return sum(
        fold_result_is_valid(Path(run_dir) / f"repetition_{repetition}" / f"fold_{fold}")
        for repetition in range(PAIRED_REPETITIONS)
        for fold in range(PAIRED_FOLDS)
    )


def _request_metadata(
    source_run: Path,
    checkpoints: dict[tuple[int, int], Path],
    source_name: str,
    target_name: str,
    model: str,
    seed: int,
    source_data_dir: Path,
    target_data_dir: Path,
) -> dict[str, Any]:
    return {
        "protocol": PAIRED_PROTOCOL,
        "source_training_run": str(Path(source_run).resolve()),
        "source_dataset": source_name,
        "target_dataset": target_name,
        "model": model,
        "seed": seed,
        "source_seed_status": "explicit_user_supplied; not present in source artifacts",
        "source_data_dir": str(Path(source_data_dir).resolve()),
        "target_data_dir": str(Path(target_data_dir).resolve()),
        "checkpoints": {
            f"repetition_{repetition}/fold_{fold}": str(path.resolve())
            for (repetition, fold), path in sorted(checkpoints.items())
        },
    }


def prepare_paired_run(
    run_dir: Path,
    source_run: Path,
    checkpoints: dict[tuple[int, int], Path],
    source_name: str,
    target_name: str,
    model: str,
    seed: int,
    source_data_dir: Path,
    target_data_dir: Path,
) -> dict[str, Any]:
    """Create or validate a resumable paired output directory."""
    run_dir = Path(run_dir)
    request = _request_metadata(
        source_run,
        checkpoints,
        source_name,
        target_name,
        model,
        seed,
        source_data_dir,
        target_data_dir,
    )
    metadata_path = run_dir / "paired_eval_metadata.json"
    if run_dir.exists() and any(run_dir.iterdir()):
        if not metadata_path.is_file():
            raise ValueError(f"Existing paired output has no provenance metadata: {run_dir}")
        existing = json.loads(metadata_path.read_text())
        if {key: existing.get(key) for key in request} != request:
            raise ValueError(f"Existing paired output metadata does not match the requested run: {run_dir}")
        return existing

    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **request,
        "evaluation_timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "in_progress",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return metadata


def validate_completed_paired_run(run_dir: Path) -> None:
    count = paired_result_count(run_dir)
    if count != PAIRED_REPETITIONS * PAIRED_FOLDS:
        raise ValueError(f"Paired evaluation has {count}/25 valid fold results in {run_dir}")


def mark_paired_complete(run_dir: Path) -> None:
    metadata_path = Path(run_dir) / "paired_eval_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update(
        {
            "status": "complete",
            "completed_fold_count": PAIRED_REPETITIONS * PAIRED_FOLDS,
            "completed_timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

