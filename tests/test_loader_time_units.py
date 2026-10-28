import pytest

pl = pytest.importorskip("polars")
pytest.importorskip("gin")

from icu_benchmarks.data.constants import DataSegment, DataSplit
from icu_benchmarks.data.loader import CommonPolarsDataset


@pytest.mark.parametrize("step_minutes", [60, 15, 5])
def test_duration_sequence_is_exposed_as_fractional_hours(step_minutes: int) -> None:
    features = pl.DataFrame(
        {
            "stay_id": [1, 1],
            "time": [0, step_minutes * 60 * 1000],
            "value": [1.0, 2.0],
        }
    ).with_columns(pl.col("time").cast(pl.Duration("ms")))
    outcomes = pl.DataFrame({"stay_id": [1], "label": [0.0]})
    data = {
        split: {
            DataSegment.features: features,
            DataSegment.outcome: outcomes,
        }
        for split in (DataSplit.train, DataSplit.val, DataSplit.test)
    }

    dataset = CommonPolarsDataset(
        data,
        vars={"GROUP": "stay_id", "SEQUENCE": "time", "LABEL": "label"},
    )

    assert dataset.row_indicators["time"].to_list() == pytest.approx([0.0, step_minutes / 60.0])
    assert "time" not in dataset.features_df.columns
