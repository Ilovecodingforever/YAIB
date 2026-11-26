"""Reference outcomes control only classification split membership."""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.model_selection import StratifiedKFold

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment as Segment, DataSplit as Split
from icu_benchmarks.data.split_process_data import make_single_split, preprocess_data


FILES = {Segment.dynamic: "dyn.parquet", Segment.outcome: "outc.parquet", Segment.static: "sta.parquet"}
VARS = {"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label", "DYNAMIC": ["x"], "STATIC": ["age"]}


def write_dataset(path: Path, labels: np.ndarray) -> None:
    path.mkdir(parents=True, exist_ok=True)
    ids = np.arange(len(labels), dtype=np.int64)
    pl.DataFrame({"stay_id": ids, "time": np.zeros(len(ids)), "x": ids.astype(float)}).write_parquet(path / "dyn.parquet")
    pl.DataFrame({"stay_id": ids, "label": labels}).write_parquet(path / "outc.parquet")
    pl.DataFrame({"stay_id": ids, "age": ids.astype(float)}).write_parquet(path / "sta.parquet")


def split_ids(data, split):
    return set(data[split][Segment.outcome]["stay_id"].to_list())


def preprocess(path, reference=None, **kwargs):
    return preprocess_data(
        path,
        file_names=dict(FILES),
        vars=dict(VARS),
        required_segments=list(FILES),
        cv_repetitions=kwargs.pop("cv_repetitions", 5),
        cv_folds=kwargs.pop("cv_folds", 5),
        split_reference_outcomes_path=reference,
        **kwargs,
    )[0]


def test_reference_matches_all_five_by_five_splits_and_preserves_labels(tmp_path):
    ids = np.arange(50, dtype=np.int64)
    reference_labels = (ids < 25).astype(np.int8)
    h12_labels = (ids % 3 == 0).astype(np.int8)
    h12 = tmp_path / "h12"
    h48 = tmp_path / "h48"
    reference = tmp_path / "h48_reference.parquet"
    write_dataset(h12, h12_labels)
    write_dataset(h48, reference_labels)
    for dataset in (h12, h48):
        outcome = pl.read_parquet(dataset / "outc.parquet").sample(fraction=1.0, seed=23)
        outcome.write_parquet(dataset / "outc.parquet")
    pl.DataFrame({"stay_id": ids, "label": reference_labels}).sample(fraction=1.0, seed=17).write_parquet(reference)

    for repetition in range(5):
        for fold in range(5):
            h12_split = preprocess(h12, reference, repetition_index=repetition, fold_index=fold)
            h48_split = preprocess(h48, reference, repetition_index=repetition, fold_index=fold)
            for split in (Split.train, Split.val, Split.test):
                assert split_ids(h12_split, split) == split_ids(h48_split, split)
            memberships = [split_ids(h12_split, split) for split in (Split.train, Split.val, Split.test)]
            assert set.union(*memberships) == set(ids)
            assert not memberships[0] & memberships[1]
            assert not memberships[0] & memberships[2]
            assert not memberships[1] & memberships[2]
            outer = list(StratifiedKFold(5, shuffle=True, random_state=42).split(ids, reference_labels))[repetition]
            dev, test = outer
            train, val = list(
                StratifiedKFold(5, shuffle=True, random_state=42).split(dev, reference_labels[dev])
            )[fold]
            expected = {
                Split.train: set(ids[dev[train]]),
                Split.val: set(ids[dev[val]]),
                Split.test: set(ids[test]),
            }
            assert {split: split_ids(h12_split, split) for split in expected} == expected
            assert h12_split[Split.train][Segment.outcome].sort("stay_id")["label"].to_list() == [
                int(h12_labels[i]) for i in sorted(split_ids(h12_split, Split.train))
            ]


def test_default_splitting_is_unchanged_and_reproducible(tmp_path):
    ids = np.arange(40, dtype=np.int64)
    raw = {
        Segment.outcome: pl.DataFrame({"stay_id": ids, "label": (ids % 4 == 0)}),
        Segment.dynamic: pl.DataFrame({"stay_id": ids, "time": np.zeros(40), "x": ids.astype(float)}),
    }
    first = make_single_split(raw, VARS, 5, 0, 5, 0)
    again = make_single_split(raw, VARS, 5, 0, 5, 0)
    assert {split: split_ids(first, split) for split in (Split.train, Split.val, Split.test)} == {
        split: split_ids(again, split) for split in (Split.train, Split.val, Split.test)
    }


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.slice(1), "missing reference IDs"),
        (lambda frame: frame.with_columns(pl.col("stay_id").replace(0, 999)), "extra reference IDs"),
        (lambda frame: pl.concat([frame, frame.head(1)]), "duplicate stay_id"),
    ],
)
def test_reference_id_mismatches_are_rejected(tmp_path, mutate, message):
    ids = np.arange(20, dtype=np.int64)
    data_path = tmp_path / "data"
    reference_path = tmp_path / "reference.parquet"
    write_dataset(data_path, ids % 2)
    reference = pl.DataFrame({"stay_id": ids, "label": ids % 2})
    mutate(reference).write_parquet(reference_path)
    with pytest.raises((ValueError, pl.exceptions.DuplicateError), match=message):
        preprocess(data_path, reference_path, cv_repetitions=2, cv_folds=2)


@pytest.mark.parametrize("labels", [[None] + [0] * 9, [0.1] * 10])
def test_invalid_reference_labels_are_rejected(tmp_path, labels):
    ids = np.arange(10, dtype=np.int64)
    data_path = tmp_path / "data"
    reference_path = tmp_path / "reference.parquet"
    write_dataset(data_path, ids % 2)
    pl.DataFrame({"stay_id": ids, "label": labels}).write_parquet(reference_path)
    with pytest.raises(ValueError, match="reference.*(missing|continuous)"):  # noqa: RUF001
        preprocess(data_path, reference_path, cv_repetitions=2, cv_folds=2)


def test_reference_cache_identity_changes_with_reference_content(tmp_path):
    ids = np.arange(20, dtype=np.int64)
    data_path = tmp_path / "data"
    first_reference = tmp_path / "reference_1.parquet"
    second_reference = tmp_path / "reference_2.parquet"
    write_dataset(data_path, ids % 2)
    pl.DataFrame({"stay_id": ids, "label": ids % 2}).write_parquet(first_reference)
    pl.DataFrame({"stay_id": ids, "label": (ids + 1) % 2}).write_parquet(second_reference)

    preprocess(data_path, first_reference, generate_cache=True, cv_repetitions=2, cv_folds=2)
    preprocess(data_path, second_reference, generate_cache=True, cv_repetitions=2, cv_folds=2)
    assert len(list((data_path / "cache").iterdir())) == 2


def test_reference_manifest_is_written_to_the_requested_log_path(tmp_path):
    ids = np.arange(20, dtype=np.int64)
    data_path = tmp_path / "data"
    reference_path = tmp_path / "reference.parquet"
    manifest_path = tmp_path / "logs" / "repetition_0" / "fold_0" / "split_manifest.parquet"
    write_dataset(data_path, ids % 2)
    pl.DataFrame({"stay_id": ids, "label": ids % 2}).write_parquet(reference_path)

    preprocess(data_path, reference_path, split_manifest_path=manifest_path, cv_repetitions=2, cv_folds=2)
    manifest = pl.read_parquet(manifest_path)
    assert set(manifest.columns) == {
        "seed",
        "repetition",
        "fold",
        "split",
        "stay_id",
        "reference_path",
        "reference_sha256",
    }
    assert manifest.height == len(ids)
    assert not (data_path / "split_manifest.parquet").exists()


def test_complete_train_reference_split_uses_target_labels(tmp_path):
    ids = np.arange(20, dtype=np.int64)
    data_path = tmp_path / "data"
    reference_path = tmp_path / "reference.parquet"
    target_labels = (ids % 3 == 0).astype(int)
    write_dataset(data_path, target_labels)
    pl.DataFrame({"stay_id": ids, "label": ids % 2}).write_parquet(reference_path)

    data = preprocess(data_path, reference_path, complete_train=True, train_size=0.8, cv_repetitions=2, cv_folds=2)
    assert split_ids(data, Split.test) == split_ids(data, Split.val)
    for split in (Split.train, Split.val):
        observed = data[split][Segment.outcome].sort("stay_id")["label"].to_list()
        expected = [int(target_labels[i]) for i in sorted(split_ids(data, split))]
        assert observed == expected


def test_pandas_reference_split_preserves_target_labels():
    ids = np.arange(40, dtype=np.int64)
    target = (ids % 3 == 0).astype(int)
    reference = pd.DataFrame({"stay_id": ids, "label": (ids < 20).astype(int)})
    raw = {
        Segment.outcome: pd.DataFrame({"stay_id": ids, "label": target}),
        Segment.dynamic: pd.DataFrame({"stay_id": ids, "time": np.zeros(40), "x": ids.astype(float)}),
    }
    split = make_single_split(raw, VARS, 5, 0, 5, 0, split_reference_outcomes=reference, polars=False)
    for name in (Split.train, Split.val, Split.test):
        observed = split[name][Segment.outcome].set_index("stay_id")["label"]
        assert observed.to_dict() == {int(i): int(target[i]) for i in observed.index}


def test_reference_path_is_rejected_for_regression(tmp_path):
    ids = np.arange(10, dtype=np.int64)
    data_path = tmp_path / "data"
    reference_path = tmp_path / "reference.parquet"
    write_dataset(data_path, ids)
    pl.DataFrame({"stay_id": ids, "label": ids % 2}).write_parquet(reference_path)
    with pytest.raises(ValueError, match="only for classification"):
        preprocess(data_path, reference_path, runmode=RunMode.regression, cv_repetitions=2, cv_folds=2)
