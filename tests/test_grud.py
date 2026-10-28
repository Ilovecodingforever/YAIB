import numpy as np
import gin
import pandas as pd
import polars as pl
import pytest
import torch
from datetime import datetime, timezone, timedelta

from icu_benchmarks.constants import RunMode
from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.loader import GRUDPredictionPandasDataset, GRUDPredictionPolarsDataset, _grud_elapsed_time
from icu_benchmarks.models.dl_models.grud import GRUDNet, _GRUDCell


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
    second_times = [times[0], times[min(1, len(times) - 1)]]
    return pl.DataFrame(
        {
            "stay_id": [1] * len(times) + [2, 2],
            "time": times + second_times,
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
    dataset = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
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
    train = GRUDPredictionPolarsDataset(data, split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
    val = GRUDPredictionPolarsDataset(
        data,
        split=DataSplit.val,
        vars=VARS,
        time_unit="hours",
        reference_means=train.get_reference_means(),
        ram_cache=False,
    )
    assert train.get_reference_means().tolist() == pytest.approx([14.0 / 3.0, 38.0 / 3.0])
    assert val.get_reference_means().tolist() == pytest.approx([14.0 / 3.0, 38.0 / 3.0])

    no_observations = train_features.with_columns(pl.lit(1.0).alias("MissingIndicator_x"))
    no_obs_data = _data(no_observations)
    no_obs = GRUDPredictionPolarsDataset(no_obs_data, split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
    assert no_obs.get_reference_means()[0] == 0.0


def test_grud_frequency_changes_mask_and_elapsed_inputs():
    regular = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
    sparse_features = _features(times=(0.0, 2.0, 4.0), x_missing=(0.0, 1.0, 0.0))
    sparse = GRUDPredictionPolarsDataset(
        _data(sparse_features), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False
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
        time_unit="hours",
        ram_cache=False,
    )
    assert hidden[0][0][1][:, 0].tolist() == [1.0, 1.0, 1.0]
    assert hidden[0][0][2][:, 0].tolist() == pytest.approx([0.0, 2.0, 2.0])


def test_grud_subhour_numeric_timestamps_are_hours():
    features = _features(times=(0.0, 0.25, 1.0), x_missing=(0.0, 1.0, 0.0))
    dataset = GRUDPredictionPolarsDataset(_data(features), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
    elapsed = dataset[0][0][2]
    assert elapsed[:, 0].tolist() == pytest.approx([0.0, 0.25, 1.0])


def test_grud_polars_duration_timestamps_are_converted_to_hours():
    features = _features(times=(0.0, 15.0 * 60.0 * 1000.0, 60.0 * 60.0 * 1000.0), x_missing=(0.0, 1.0, 0.0))
    features = features.with_columns(pl.col("time").cast(pl.Duration("ms")))
    dataset = GRUDPredictionPolarsDataset(_data(features), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
    elapsed = dataset[0][0][2]
    assert elapsed[:, 0].tolist() == pytest.approx([0.0, 0.25, 1.0])


@pytest.mark.parametrize(
    ("unit", "times"),
    [
        ("hours", (10.0, 10.25, 11.0)),
        ("minutes", (600.0, 615.0, 660.0)),
        ("seconds", (36000.0, 36900.0, 39600.0)),
    ],
)
def test_grud_numeric_timestamp_units_are_explicit_and_equivalent(unit, times):
    dataset = GRUDPredictionPolarsDataset(
        _data(_features(times=times)), split=DataSplit.train, vars=VARS, time_unit=unit, ram_cache=False
    )
    assert dataset[0][0][2][:, 0].tolist() == pytest.approx([0.0, 0.25, 1.0])


def test_grud_elapsed_time_accumulates_across_consecutive_missing_rows():
    elapsed = _grud_elapsed_time(
        np.array([100.0, 115.0, 130.0, 160.0]),
        np.array([[1.0], [0.0], [0.0], [1.0]]),
        "minutes",
        stay_id=7,
    )
    assert elapsed[:, 0].tolist() == pytest.approx([0.0, 0.25, 0.5, 1.0])


def test_grud_pandas_timedeltas_and_absolute_datetimes_match_polars():
    relative = [0.0, 0.25, 1.0]
    polars_features = _features(times=relative)
    pandas_features = pd.DataFrame(polars_features.to_dict(as_series=False))
    pandas_features["time"] = pd.to_timedelta([0, 15, 60, 0, 15], unit="m")
    pandas_data = {
        split: {
            DataSegment.features: pandas_features.copy(),
            DataSegment.outcome: pd.DataFrame({"stay_id": [1, 2], "label": [0.0, 1.0]}),
        }
        for split in (DataSplit.train, DataSplit.val, DataSplit.test)
    }
    pandas_dataset = GRUDPredictionPandasDataset(
        pandas_data, split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False
    )
    assert pandas_dataset[0][0][2][:, 0].tolist() == pytest.approx([0.0, 0.25, 1.0])

    absolute = [pd.Timestamp("2024-01-01T00:00:00"), pd.Timestamp("2024-01-01T00:15:00"), pd.Timestamp("2024-01-01T01:00:00")]
    absolute_features = _features(times=absolute)
    absolute_dataset = GRUDPredictionPolarsDataset(
        _data(absolute_features), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False
    )
    assert absolute_dataset[0][0][2][:, 0].tolist() == pytest.approx([0.0, 0.25, 1.0])

    absolute_pandas = pandas_features.copy()
    absolute_pandas["time"] = [
        pd.Timestamp("2024-01-01T00:00:00"),
        pd.Timestamp("2024-01-01T00:15:00"),
        pd.Timestamp("2024-01-01T01:00:00"),
        pd.Timestamp("2024-01-01T00:00:00"),
        pd.Timestamp("2024-01-01T00:15:00"),
    ]
    absolute_pandas_data = {
        split: {
            DataSegment.features: absolute_pandas.copy(),
            DataSegment.outcome: pd.DataFrame({"stay_id": [1, 2], "label": [0.0, 1.0]}),
        }
        for split in (DataSplit.train, DataSplit.val, DataSplit.test)
    }
    absolute_pandas_dataset = GRUDPredictionPandasDataset(
        absolute_pandas_data, split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False
    )
    assert absolute_pandas_dataset[0][0][2][:, 0].tolist() == pytest.approx([0.0, 0.25, 1.0])


def test_grud_numeric_time_unit_is_required():
    with pytest.raises(RuntimeError, match="time_unit"):
        GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, ram_cache=False)


def test_grud_rejects_invalid_or_decreasing_timestamps_and_allows_duplicates():
    mask = np.ones((3, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="decrease"):
        _grud_elapsed_time(np.array([0.0, 2.0, 1.0]), mask, "hours", stay_id=3)
    with pytest.raises(ValueError, match="nonfinite"):
        _grud_elapsed_time(np.array([0.0, np.inf, 1.0]), mask, "hours", stay_id=3)
    duplicate = _grud_elapsed_time(np.array([0.0, 0.0, 1.0]), mask, "hours", stay_id=3)
    assert duplicate[:, 0].tolist() == pytest.approx([0.0, 0.0, 1.0])


def test_grud_timezone_aware_datetimes_are_normalized_before_subtraction():
    timestamps = np.array(
        [
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 1, 15, tzinfo=timezone(timedelta(hours=1))),
            datetime(2024, 1, 1, 1, tzinfo=timezone.utc),
        ],
        dtype=object,
    )
    elapsed = _grud_elapsed_time(timestamps, np.ones((3, 1), dtype=np.float32), "hours")
    assert elapsed[:, 0].tolist() == pytest.approx([0.0, 0.25, 0.75])


def test_grud_candidate_and_final_update_follow_paper_equations():
    cell = _GRUDCell(input_dim=1, hidden_dim=1)
    with torch.no_grad():
        cell.input_projection.weight.zero_()
        cell.input_projection.bias.copy_(torch.tensor([0.0, 0.0, 0.2]))
        cell.hidden_projection.weight.zero_()
        cell.mask_projection.weight.zero_()
        cell.candidate_hidden_projection.weight.fill_(2.0)
        cell.input_decay_weight.zero_()
        cell.input_decay_bias.zero_()
        cell.hidden_decay_weight.zero_()
        cell.hidden_decay_bias.fill_(0.5)

    output = cell(
        values=torch.zeros(1, 1, 1),
        observed_mask=torch.zeros(1, 1, 1),
        elapsed_time=torch.ones(1, 1, 1),
        hidden=torch.ones(1, 1),
        reference_means=torch.zeros(1),
        sequence_mask=torch.ones(1, 1, dtype=torch.bool),
    )
    decayed_hidden = torch.exp(torch.tensor(-0.5))
    candidate = torch.tanh(torch.tensor(0.2) + 2.0 * 0.5 * decayed_hidden)
    expected = 0.5 * decayed_hidden + 0.5 * candidate
    assert output.item() == pytest.approx(expected.item())


def test_grud_missing_input_decays_toward_training_mean():
    cell = _GRUDCell(input_dim=1, hidden_dim=1)
    with torch.no_grad():
        cell.input_projection.weight.zero_()
        cell.input_projection.weight[2, 0] = 1.0
        cell.input_projection.bias.copy_(torch.tensor([0.0, 20.0, 0.0]))
        cell.hidden_projection.weight.zero_()
        cell.mask_projection.weight.zero_()
        cell.candidate_hidden_projection.weight.zero_()
        cell.input_decay_weight.fill_(1.0)
        cell.input_decay_bias.zero_()
        cell.hidden_decay_weight.zero_()
        cell.hidden_decay_bias.zero_()

    output = cell(
        values=torch.tensor([[[2.0], [2.0]]]),
        observed_mask=torch.tensor([[[1.0], [0.0]]]),
        elapsed_time=torch.tensor([[[0.0], [1.0]]]),
        hidden=torch.zeros(1, 1),
        reference_means=torch.tensor([0.5]),
        sequence_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    expected_input = torch.exp(torch.tensor(-1.0)) * 2.0 + (1.0 - torch.exp(torch.tensor(-1.0))) * 0.5
    assert output[0, 1, 0].item() == pytest.approx(torch.tanh(expected_input).item(), abs=1e-5)


def test_grud_decay_parameters_receive_gradients_and_update():
    torch.manual_seed(7)
    cell = _GRUDCell(input_dim=2, hidden_dim=3)
    optimizer = torch.optim.SGD(cell.parameters(), lr=0.1)
    before = {
        name: parameter.detach().clone()
        for name, parameter in cell.named_parameters()
        if "decay" in name
    }
    output = cell(
        values=torch.tensor([[[2.0, -1.0], [2.0, -1.0]]]),
        observed_mask=torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]),
        elapsed_time=torch.tensor([[[0.0, 0.0], [1.0, 0.5]]]),
        hidden=torch.zeros(1, 3),
        reference_means=torch.tensor([0.5, -0.5]),
        sequence_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    output.sum().backward()
    for name, parameter in cell.named_parameters():
        if "decay" in name:
            assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    optimizer.step()
    for name, parameter in cell.named_parameters():
        if "decay" in name:
            assert not torch.equal(before[name], parameter), name


def test_grud_output_shape_and_padding_state_are_stable():
    dataset = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
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
    dataset = GRUDPredictionPolarsDataset(_data(), split=DataSplit.train, vars=VARS, time_unit="hours", ram_cache=False)
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
    train_dir.mkdir()
    gin.clear_config()
    gin.bind_parameter("GRUDPredictionPolarsDataset.vars", VARS)
    gin.bind_parameter("GRUDPredictionPolarsDataset.time_unit", "hours")
    gin.bind_parameter("GRUDNet.hidden_dim", 4)
    gin.bind_parameter("GRUDNet.layer_dim", 1)
    gin.bind_parameter("GRUDNet.num_classes", 2)
    gin.bind_parameter("DLPredictionWrapper.loss", torch.nn.functional.cross_entropy)
    try:
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

        eval_dir = tmp_path / "eval"
        eval_dir.mkdir()
        eval_score = train_common(
            data,
            log_dir=eval_dir,
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
    finally:
        gin.clear_config()
