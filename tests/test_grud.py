import numpy as np
import polars as pl
import pytest
import torch

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.loader import GRUDPredictionPolarsDataset
from icu_benchmarks.models.dl_models.grud import GRUDNet


VARS = {
    "GROUP": "stay_id",
    "SEQUENCE": "time",
    "LABEL": "label",
    "DYNAMIC": ["x", "y"],
    "STATIC": ["age"],
}


def _features(times=(0.0, 1.0, 4.0), x_missing=(0.0, 1.0, 0.0)):
    times = list(times)
    x_missing = list(x_missing)
    return pl.DataFrame(
        {
            "stay_id": [1] * len(times) + [2, 2],
            "time": times + [0.0, 2.0],
            "x": [2.0, 3.0, 5.0][: len(times)] + [7.0, 8.0],
            "y": [10.0, 11.0, 12.0][: len(times)] + [13.0, 14.0],
            "MissingIndicator_x": x_missing + [0.0, 1.0],
            "MissingIndicator_y": [1.0, 0.0, 1.0][: len(times)] + [0.0, 0.0],
            "age": [60.0] * len(times) + [70.0, 70.0],
            "MissingIndicator_age": [0.0] * (len(times) + 2),
        }
    )


def _data(features=None):
    features = _features() if features is None else features
    outcome = pl.DataFrame({"stay_id": [1, 2], "label": [0.0, 1.0]})
    return {
        split: {DataSegment.features: features, DataSegment.outcome: outcome}
        for split in (DataSplit.train, DataSplit.val, DataSplit.test)
    }


def test_grud_dataset_exposes_observed_masks_and_elapsed_times():
    dataset = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, ram_cache=False)
    inputs, _, prediction_mask = dataset[0]
    values, observed, elapsed, static, sequence_mask = inputs

    assert values.shape == (3, 2)
    assert observed[:, 0].tolist() == [1.0, 0.0, 1.0]
    assert observed[:, 1].tolist() == [0.0, 1.0, 0.0]
    assert elapsed[:, 0].tolist() == pytest.approx([0.0, 1.0, 4.0])
    assert elapsed[:, 1].tolist() == pytest.approx([0.0, 1.0, 3.0])
    assert static.tolist() == [60.0, 0.0]
    assert sequence_mask.tolist() == [True, True, True]
    assert prediction_mask.tolist() == [False, False, True]


def test_grud_means_are_training_only_and_missing_training_variables_are_explicit():
    train_features = _features()
    val_features = _features().with_columns(pl.lit(100.0).alias("x"))
    data = _data(train_features)
    data[DataSplit.val][DataSegment.features] = val_features
    train = GRUDPredictionPolarsDataset(data, split=DataSplit.train, vars=VARS, ram_cache=False)
    val = GRUDPredictionPolarsDataset(
        data,
        split=DataSplit.val,
        vars=VARS,
        reference_means=train.get_reference_means(),
        ram_cache=False,
    )
    assert train.get_reference_means().tolist() == pytest.approx([14.0 / 3.0, 38.0 / 3.0])
    assert val.get_reference_means().tolist() == pytest.approx([14.0 / 3.0, 38.0 / 3.0])

    no_observations = train_features.with_columns(pl.lit(1.0).alias("MissingIndicator_x"))
    no_obs_data = _data(no_observations)
    no_obs = GRUDPredictionPolarsDataset(no_obs_data, split=DataSplit.train, vars=VARS, ram_cache=False)
    assert no_obs.get_reference_means()[0] == 0.0


def test_grud_frequency_changes_mask_and_elapsed_inputs():
    regular = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, ram_cache=False)
    sparse_features = _features(times=(0.0, 2.0, 4.0), x_missing=(0.0, 1.0, 0.0))
    sparse = GRUDPredictionPolarsDataset(
        _data(sparse_features), split=DataSplit.train, vars=VARS, ram_cache=False
    )
    regular_inputs = regular[0][0]
    sparse_inputs = sparse[0][0]
    assert regular_inputs[1][:, 0].tolist() == sparse_inputs[1][:, 0].tolist()
    assert regular_inputs[2][:, 0].tolist() == pytest.approx([0.0, 1.0, 4.0])
    assert sparse_inputs[2][:, 0].tolist() == pytest.approx([0.0, 2.0, 4.0])
    hidden = GRUDPredictionPolarsDataset(
        _data(_features(times=(0.0, 2.0, 4.0), x_missing=(0.0, 0.0, 0.0))),
        split=DataSplit.train,
        vars=VARS,
        ram_cache=False,
    )
    assert hidden[0][0][1][:, 0].tolist() == [1.0, 1.0, 1.0]
    assert hidden[0][0][2][:, 0].tolist() == pytest.approx([0.0, 2.0, 2.0])


def test_grud_output_shape_and_padding_state_are_stable():
    dataset = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, ram_cache=False)
    first_inputs = dataset[0][0]
    second_inputs = dataset[1][0]
    model = GRUDNet(
        input_size=tuple(value.shape for value in first_inputs),
        hidden_dim=4,
        layer_dim=2,
        num_classes=2,
        dropout=0.0,
        reference_means=dataset.get_reference_means(),
        run_mode=RunMode.classification,
    )
    model.eval()
    with torch.no_grad():
        output = model([value.unsqueeze(0) for value in second_inputs])
    assert output.shape == (1, 3, 2)
    assert hasattr(model, "logit")
    assert torch.allclose(output[:, 2], output[:, 1])


def test_grud_checkpoint_persists_reference_means():
    dataset = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, ram_cache=False)
    inputs = dataset[0][0]
    model = GRUDNet(
        input_size=tuple(value.shape for value in inputs),
        hidden_dim=3,
        layer_dim=1,
        num_classes=2,
        reference_means=dataset.get_reference_means(),
    )
    checkpoint = {}
    model.on_save_checkpoint(checkpoint)
    reloaded = GRUDNet(
        input_size=tuple(value.shape for value in inputs),
        hidden_dim=3,
        layer_dim=1,
        num_classes=2,
    )
    reloaded.on_load_checkpoint(checkpoint)
    assert torch.equal(model.reference_means, reloaded.reference_means)


def test_grud_end_to_end_train_and_evaluate(tmp_path):
    from icu_benchmarks.models.train import train_common

    data = _data()
    train_dir = tmp_path / "train"
    score = train_common(
        data,
        log_dir=train_dir,
        model=GRUDNet,
        mode=RunMode.classification,
        weight="balanced",
        epochs=1,
        batch_size=2,
        patience=1,
        cpu=True,
        num_workers=0,
        ram_cache=False,
        verbose=False,
    )
    assert np.isfinite(score)
    assert (train_dir / "model.ckpt").exists()

    eval_score = train_common(
        data,
        log_dir=tmp_path / "eval",
        model=GRUDNet,
        mode=RunMode.classification,
        weight="balanced",
        eval_only=True,
        load_weights=True,
        source_dir=train_dir,
        epochs=1,
        batch_size=2,
        cpu=True,
        num_workers=0,
        ram_cache=False,
        verbose=False,
    )
    assert np.isfinite(eval_score)
