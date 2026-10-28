import logging
import warnings
from datetime import date, datetime, timedelta
from typing import List, Optional, Union

import gin
import numpy as np
import pandas as pd
import polars as pl
from pandas import DataFrame
from torch import Tensor, cat, float32, from_numpy
from torch.utils.data import Dataset

from icu_benchmarks.imputation.amputations import ampute_data

from .constants import DataSegment as DataSegment
from .constants import DataSplit as DataSplit

GRUD_TIME_UNIT = "hours"
_GRUD_TIME_FACTORS = {"hours": 1.0, "minutes": 1.0 / 60.0, "seconds": 1.0 / 3600.0}


def _grud_validate_time_unit(time_unit: str) -> None:
    if time_unit not in _GRUD_TIME_FACTORS:
        raise ValueError(
            "GRU-D numeric_time_unit must be one of 'hours', 'minutes', or 'seconds'; "
            f"got {time_unit!r}."
        )


def _grud_timestamp_hours(sequence, numeric_time_unit: str, stay_id=None) -> np.ndarray:
    """Convert one stay's timestamps to relative hours without losing precision.

    Numeric timestamps require ``numeric_time_unit``; no unit is inferred from
    magnitude. Duplicate timestamps are allowed, retain stable input order,
    and produce a zero gap. Mixed timezone-aware and naive datetimes are
    rejected because their ordering is ambiguous.
    """
    _grud_validate_time_unit(numeric_time_unit)

    raw = np.asarray(sequence)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"GRU-D timestamps for stay {stay_id!r} must be a non-empty one-dimensional sequence.")

    non_null = [value for value in raw.tolist() if value is not None and not pd.isna(value)]
    if len(non_null) != raw.size:
        raise ValueError(f"GRU-D timestamps for stay {stay_id!r} contain missing values.")

    is_timedelta = np.issubdtype(raw.dtype, np.timedelta64) or all(
        isinstance(value, (timedelta, np.timedelta64, pd.Timedelta)) for value in non_null
    )
    is_datetime = np.issubdtype(raw.dtype, np.datetime64) or all(
        isinstance(value, (datetime, date, np.datetime64, pd.Timestamp)) for value in non_null
    )
    if is_timedelta:
        try:
            converted = pd.to_timedelta(raw)
            relative = converted - converted[0]
            hours = np.asarray(relative.total_seconds(), dtype=np.float64) / 3600.0
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Could not convert GRU-D timedeltas for stay {stay_id!r}: {error}") from error
    elif is_datetime:
        aware = [pd.Timestamp(value).tzinfo is not None for value in non_null]
        if any(aware) and not all(aware):
            raise ValueError(f"GRU-D timestamps for stay {stay_id!r} mix timezone-aware and naive datetimes.")
        try:
            converted = pd.to_datetime(raw, utc=True)
            relative = converted - converted[0]
            hours = np.asarray(relative.total_seconds(), dtype=np.float64) / 3600.0
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Could not convert GRU-D datetimes for stay {stay_id!r}: {error}") from error
    else:
        try:
            numeric = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Unsupported GRU-D timestamp format for stay {stay_id!r}; use numeric values with an explicit "
                "numeric_time_unit, Polars/pandas durations, or datetimes."
            ) from error
        if not np.isfinite(numeric).all():
            raise ValueError(f"GRU-D timestamps for stay {stay_id!r} contain nonfinite values.")
        # Subtract before scaling/casting: this preserves subhour gaps near a modern epoch.
        hours = (numeric - numeric[0]) * _GRUD_TIME_FACTORS[numeric_time_unit]

    if not np.isfinite(hours).all():
        raise ValueError(f"GRU-D timestamps for stay {stay_id!r} contain nonfinite values.")
    if np.any(np.diff(hours) < 0):
        raise ValueError(f"GRU-D timestamps for stay {stay_id!r} decrease; timestamps must be nondecreasing.")
    return hours.astype(np.float32)


def _grud_elapsed_time(
    sequence: np.ndarray,
    observed_mask: np.ndarray,
    numeric_time_unit: str,
    stay_id=None,
) -> np.ndarray:
    """Return per-variable hours since the previous observed value.

    ``observed_mask`` uses YAIB's polarity: one means the measurement was
    observed. Missing rows leave the last-observation timestamp unchanged.
    """
    observed_mask = np.asarray(observed_mask, dtype=np.float32)
    if observed_mask.ndim != 2:
        raise ValueError("GRU-D observed_mask must be a two-dimensional array.")
    timestamps = _grud_timestamp_hours(sequence, numeric_time_unit, stay_id)
    if len(timestamps) != observed_mask.shape[0]:
        raise ValueError(
            f"GRU-D timestamps and observed_mask disagree for stay {stay_id!r}: "
            f"{len(timestamps)} timestamps versus {observed_mask.shape[0]} rows."
        )
    elapsed = np.zeros_like(observed_mask, dtype=np.float32)
    last_observation = np.zeros(observed_mask.shape[1], dtype=np.float32)
    for timestep, timestamp in enumerate(timestamps):
        elapsed[timestep] = timestamp - last_observation
        last_observation[observed_mask[timestep] > 0.5] = timestamp
    return elapsed


def _grud_reference_means(features, vars: dict[str, Union[str, list[str]]]) -> np.ndarray:
    """Calculate means from observed training values in model/value space."""
    dynamic = list(vars.get(DataSegment.dynamic, []))
    indicator_names = [f"MissingIndicator_{name}" for name in dynamic]
    if hasattr(features, "select"):
        values = features.select(dynamic).to_numpy().astype(np.float32)
        missing = features.select(indicator_names).to_numpy().astype(np.float32)
    else:
        values = features[dynamic].to_numpy(dtype=np.float32)
        missing = features[indicator_names].to_numpy(dtype=np.float32)
    means = np.zeros(len(dynamic), dtype=np.float32)
    for index in range(len(dynamic)):
        observed = (missing[:, index] < 0.5) & np.isfinite(values[:, index])
        if observed.any():
            means[index] = values[observed, index].mean()
        else:
            logging.warning(
                "GRU-D variable %s has no observed training values; using zero as its reference mean.",
                dynamic[index],
            )
    return means


def _grud_validate_columns(features, vars: dict[str, Union[str, list[str]]]) -> tuple[list[str], list[str], list[str]]:
    columns = set(features.columns)
    dynamic = list(vars.get(DataSegment.dynamic, []))
    if not dynamic:
        raise ValueError("GRU-D requires at least one dynamic variable.")
    missing_indicators = [f"MissingIndicator_{name}" for name in dynamic]
    missing = [name for name in dynamic + missing_indicators if name not in columns]
    if missing:
        raise ValueError(
            "GRU-D requires preprocessed dynamic values and MissingIndicator columns; "
            f"missing columns: {missing}"
        )
    static = [name for name in vars.get(DataSegment.static, []) if name in columns]
    static += [
        f"MissingIndicator_{name}"
        for name in vars.get(DataSegment.static, [])
        if f"MissingIndicator_{name}" in columns
    ]
    return dynamic, missing_indicators, static


def _grud_pad_item(
    values: np.ndarray,
    observed_mask: np.ndarray,
    elapsed_time: np.ndarray,
    static: np.ndarray,
    labels: np.ndarray,
    maxlen: int,
) -> tuple[list[Tensor], Tensor, Tensor]:
    if len(labels) == 1:
        labels = np.concatenate(
            [np.full((values.shape[0] - 1, *labels.shape[1:]), np.nan), labels],
            axis=0,
        )

    length_diff = maxlen - values.shape[0]
    sequence_mask = np.ones(values.shape[0], dtype=bool)
    if length_diff > 0:
        pad_shape = (length_diff, values.shape[1])
        values = np.concatenate([values, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        observed_mask = np.concatenate([observed_mask, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        elapsed_time = np.concatenate([elapsed_time, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        labels = np.concatenate([labels, np.full((length_diff, *labels.shape[1:]), np.nan)], axis=0)
        sequence_mask = np.concatenate([sequence_mask, np.zeros(length_diff, dtype=bool)], axis=0)

    nan_mask = np.isnan(labels)
    labels[nan_mask] = -1
    invalid_rows = nan_mask if labels.ndim == 1 else nan_mask.any(axis=-1)
    prediction_mask = sequence_mask.copy()
    prediction_mask[invalid_rows] = False
    return (
        [
            from_numpy(values.astype(np.float32)),
            from_numpy(observed_mask.astype(np.float32)),
            from_numpy(elapsed_time.astype(np.float32)),
            from_numpy(static.astype(np.float32)),
            from_numpy(sequence_mask),
        ],
        from_numpy(labels.astype(np.float32)),
        from_numpy(prediction_mask),
    )


@gin.configurable("CommonPolarsDataset")
class CommonPolarsDataset(Dataset):
    def __init__(
        self,
        data: dict,
        split: str = DataSplit.train,
        vars: dict[str, Union[str, list[str]]] | object = gin.REQUIRED,
        grouping_segment: str = DataSegment.outcome,
        mps: bool = False,
        name: str = "",
    ):
        if not isinstance(vars, dict):
            raise ValueError(f"Expected vars to be of type dict, got {type(vars)} instead")
        self.split = split
        self.vars = vars
        self.grouping_df = data[split][grouping_segment]
        # Get the row indicators for the data to be able to match predicted labels
        if not isinstance(vars["SEQUENCE"], str):
            raise ValueError(f'Expected key "SEQUENCE" to be of type str, got {type(vars["SEQUENCE"])} instead')
        if "SEQUENCE" in self.vars and self.vars["SEQUENCE"] in data[split][DataSegment.features].columns:
            # We have a time series dataset
            self.row_indicators = data[split][DataSegment.features][self.vars["GROUP"], self.vars["SEQUENCE"]]

            sequence = self.vars["SEQUENCE"]
            if self.row_indicators.schema[sequence].base_type() == pl.Duration:
                self.row_indicators = self.row_indicators.with_columns(
                    (
                        pl.col(sequence).dt.total_seconds(fractional=True) / 3600.0
                    ).alias(sequence)
                )
            self.features_df = data[split][DataSegment.features]
            self.features_df = self.features_df.sort([self.vars["GROUP"], self.vars["SEQUENCE"]])
            self.features_df = self.features_df.drop(self.vars["SEQUENCE"])
        else:
            # We have a static dataset
            logging.info("Using static dataset")
            self.row_indicators = data[split][DataSegment.features][self.vars["GROUP"]]
            self.features_df = data[split][DataSegment.features]

        # order columns: index, features (alphabetically), indicator (alphabetically)
        cols = self.features_df.columns
        m_index = [self.vars["GROUP"]]
        front = sorted([c for c in cols if not c.startswith("MissingIndicator_") and c not in m_index])
        back = sorted([c for c in cols if c.startswith("MissingIndicator_") and c not in m_index])
        self.features_df = self.features_df[m_index + front + back]

        # calculate basic info for the data
        self.num_stays = self.grouping_df[self.vars["GROUP"]].unique().shape[0]
        self.maxlen = self.features_df.group_by([self.vars["GROUP"]]).len().max().item(0, 1)
        self.mps = mps
        self.name = name

    def ram_cache(self, cache: bool = True):
        self._cached_dataset = None
        if cache:
            logging.info(f"Caching {self.split} dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """
        Returns:
            number of stays in the data
        """
        return self.num_stays

    def get_feature_names(self) -> List[str]:
        # The group column is not a model input, see get_data_and_labels and _build_item
        return [c for c in self.features_df.columns if c != self.vars["GROUP"]]

    def to_tensor(self) -> tuple[Union[Tensor, np.ndarray], ...]:
        values: list[list] = []
        for entry in self:
            for i, value in enumerate(entry):
                if len(values) <= i:
                    values.append([])
                values[i].append(value.unsqueeze(0))
        return tuple(cat(value, dim=0) for value in values)


@gin.configurable("PredictionPolarsDataset")
class PredictionPolarsDataset(CommonPolarsDataset):
    """Subclass of common dataset for prediction tasks.

    Args:
        ram_cache (bool, optional): Whether the complete dataset should be stored in ram. Defaults to True.
    """

    def __init__(self, *args, ram_cache: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.outcome_df = self.grouping_df

        # One-pass partition keyed by stay id. Avoids the O(N_stays) full-frame filter
        # that previously ran inside every __getitem__.
        GROUP = self.vars["GROUP"]
        LABEL = self.vars["LABEL"]
        label_partitions = self.outcome_df.partition_by(GROUP, maintain_order=True)
        feat_partitions = self.features_df.partition_by(GROUP, maintain_order=True)
        self._stay_order = [part[GROUP][0] for part in label_partitions]
        self._feat_arrays = {
            part[GROUP][0]: part.select(pl.exclude(GROUP)).to_numpy().astype(np.float32) for part in feat_partitions
        }
        for arr in self._feat_arrays.values():
            arr.setflags(write=False)

        self._label_arrays = {part[GROUP][0]: part[LABEL].to_numpy().astype(np.float32) for part in label_partitions}
        for arr in self._label_arrays.values():
            arr.setflags(write=False)

        self.ram_cache(ram_cache)

    def _build_item(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        pad_value = 0.0
        stay_id = self._stay_order[idx]

        window = self._feat_arrays[stay_id]
        labels = self._label_arrays[stay_id].copy()  # copy to avoid in-place NaN replacement

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate(
                [np.full((window.shape[0] - 1, *labels.shape[1:]), np.nan), labels],
                axis=0,
            )

        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones(window.shape[0])

        # Padding the array to fulfill size requirement
        if length_diff > 0:
            window = np.concatenate([window, np.full((length_diff, window.shape[1]), pad_value)], axis=0)
            labels = np.concatenate([labels, np.full((length_diff, *labels.shape[1:]), pad_value)], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros(length_diff)], axis=0)

        # check for (partially) unlabeled data
        nan_mask = np.isnan(labels)
        labels[nan_mask] = -1
        invalid_rows = nan_mask if labels.ndim == 1 else nan_mask.any(axis=-1)
        pad_mask[invalid_rows] = 0

        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        data = window.astype(np.float32)

        return from_numpy(data), from_numpy(labels), from_numpy(pad_mask)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice. Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        return self._build_item(idx)

    def get_feature_names(self) -> List[str]:
        names = super().get_feature_names()
        if len(self.outcome_df) == self.num_stays:
            # get_data_and_labels drops the hour index for one-row-per-stay tasks
            names = [c for c in names if c != "counter"]
        return names

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """
        counts = self.outcome_df[self.vars["LABEL"]].value_counts(parallel=True).get_columns()[1]
        counts = counts.to_numpy()
        weights = list((1 / counts) * np.sum(counts) / counts.shape[0])
        return weights

    def get_data_and_labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A Tuple containing data points and label for the split.
        """
        labels = self.outcome_df[self.vars["LABEL"]].to_numpy().astype(np.float32)
        rep = self.features_df
        group = self.vars["GROUP"]

        if len(labels) == self.num_stays:
            # One row per stay. polars group_by does not guarantee the group order, so align the rows to the
            # label order explicitly
            rep = self.outcome_df.select(group).join(rep.group_by(group).last(), on=group, how="left")
            # The hour index from the preprocessor is a feature for the hourly tasks only
            rep = rep.drop("counter", strict=False)
            row_indicators = self.outcome_df.select(group)
        else:
            row_indicators = self.row_indicators
        # Don't use group as model feature
        rep = rep.drop(group).to_numpy().astype(np.float32)
        logging.debug(f"rep shape: {rep.shape}")
        logging.debug(f"labels shape: {labels.shape}")
        valid = np.isfinite(labels.reshape(len(labels), -1)).all(axis=1)
        return rep[valid], labels[valid], row_indicators.to_numpy()[valid]

    def to_tensor(self) -> tuple[Tensor, Tensor, Tensor]:
        data, labels, row_indicators = self.get_data_and_labels()

        return (
            from_numpy(data),
            from_numpy(labels),
            from_numpy(row_indicators),
        )


@gin.configurable("GRUDPredictionPolarsDataset")
class GRUDPredictionPolarsDataset(PredictionPolarsDataset):
    """Prediction dataset that exposes values, masks, deltas, and static context to GRU-D."""

    def ram_cache(self, cache: bool = True):
        # PredictionPolarsDataset calls this from its constructor before GRU-D arrays exist.
        self._cached_dataset = None
        if cache and hasattr(self, "_grud_dynamic_arrays"):
            logging.info(f"Caching {self.split} GRU-D dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __init__(
        self,
        *args,
        reference_means: Optional[np.ndarray] = None,
        time_unit: str = gin.REQUIRED,
        ram_cache: bool = True,
        **kwargs,
    ):
        data = args[0] if args else kwargs["data"]
        _grud_validate_time_unit(time_unit)
        super().__init__(*args, ram_cache=False, **kwargs)
        # CommonPolarsDataset converts its row indicators to hours for legacy
        # models. GRU-D uses the raw sequence column here so conversion occurs
        # exactly once in _grud_timestamp_hours.
        raw_features = data[self.split][DataSegment.features].sort(
            [self.vars["GROUP"], self.vars["SEQUENCE"]], maintain_order=True
        )
        dynamic, indicators, static = _grud_validate_columns(raw_features, self.vars)
        self._grud_dynamic_names = dynamic
        self._grud_indicator_names = indicators
        self._grud_static_names = static

        self._grud_dynamic_arrays = {}
        self._grud_mask_arrays = {}
        self._grud_static_arrays = {}
        self._grud_sequence_arrays = {}
        group = self.vars["GROUP"]
        for part in raw_features.partition_by(group, maintain_order=True):
            stay_id = part[group][0]
            self._grud_dynamic_arrays[stay_id] = part.select(dynamic).to_numpy().astype(np.float32)
            self._grud_mask_arrays[stay_id] = 1.0 - part.select(indicators).to_numpy().astype(np.float32)
            self._grud_static_arrays[stay_id] = (
                part.select(static).to_numpy()[0].astype(np.float32) if static else np.zeros(0, dtype=np.float32)
            )
            self._grud_sequence_arrays[stay_id] = part[self.vars["SEQUENCE"]].to_numpy()

        self.time_unit = time_unit

        self.reference_means = (
            np.asarray(reference_means, dtype=np.float32).copy()
            if reference_means is not None
            else _grud_reference_means(data[DataSplit.train][DataSegment.features], self.vars)
        )
        if self.reference_means.shape != (len(dynamic),):
            raise ValueError(f"Expected {len(dynamic)} GRU-D reference means, got shape {self.reference_means.shape}.")
        self.ram_cache(ram_cache)

    def get_reference_means(self) -> np.ndarray:
        return self.reference_means.copy()

    def _build_item(self, idx: int) -> tuple[list[Tensor], Tensor, Tensor]:
        stay_id = self._stay_order[idx]
        values = self._grud_dynamic_arrays[stay_id].copy()
        observed_mask = self._grud_mask_arrays[stay_id].copy()
        sequence = self._grud_sequence_arrays[stay_id]
        elapsed_time = _grud_elapsed_time(sequence, observed_mask, self.time_unit, stay_id)
        labels = self._label_arrays[stay_id].copy()
        return _grud_pad_item(
            values,
            observed_mask,
            elapsed_time,
            self._grud_static_arrays[stay_id].copy(),
            labels,
            self.maxlen,
        )


@gin.configurable("CommonPandasDataset")
class CommonPandasDataset(Dataset):
    """Common dataset: subclass of Torch Dataset that represents the data to learn on.

    Args: data: Dict of the different splits of the data. split: Either 'train','val' or 'test'. vars: Contains the names of
    columns in the data. grouping_segment: str, optional: The segment of the data contains the grouping column with only
    unique values. Defaults to Segment.outcome. Is used to calculate the number of stays in the data.
    """

    def __init__(
        self,
        data: dict,
        split: str = DataSplit.train,
        vars: dict[str, Union[str, list[str]]] | object = gin.REQUIRED,
        grouping_segment: str = DataSegment.outcome,
        mps: bool = False,
        name: str = "",
    ):
        warnings.warn(
            "CommonPandasDataset is deprecated. Use CommonPolarsDataset instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not isinstance(vars, dict):
            raise ValueError(f"Expected vars to be of type dict, got {type(vars)} instead")
        self.split = split
        self.vars = vars
        self.grouping_df = data[split][grouping_segment].set_index(self.vars["GROUP"])
        self.features_df = (
            data[split][DataSegment.features].set_index(self.vars["GROUP"]).drop(labels=self.vars["SEQUENCE"], axis=1)
        )

        # calculate basic info for the data
        self.num_stays = self.grouping_df.index.unique().shape[0]
        self.maxlen = self.features_df.groupby([self.vars["GROUP"]]).size().max()
        self.mps = mps
        self.name = name

    def ram_cache(self, cache: bool = True):
        self._cached_dataset = None
        if cache:
            logging.info(f"Caching {self.split} dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.num_stays

    def get_feature_names(self) -> list[str]:
        return self.features_df.columns

    def to_tensor(self) -> tuple[Union[Tensor, np.ndarray], ...]:
        values: list[list] = []
        for entry in self:
            for i, value in enumerate(entry):
                if len(values) <= i:
                    values.append([])
                values[i].append(value.unsqueeze(0))
        return tuple(cat(value, dim=0) for value in values)


@gin.configurable("PredictionPandasDataset")
class PredictionPandasDataset(CommonPandasDataset):
    """Subclass of common dataset for prediction tasks.

    Args:
        ram_cache (bool, optional): Whether the complete dataset should be stored in ram. Defaults to True.
    """

    def __init__(self, *args, ram_cache: bool = True, **kwargs):
        super().__init__(*args, grouping_segment=DataSegment.outcome, **kwargs)
        self.outcome_df = self.grouping_df
        self.ram_cache(ram_cache)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice. Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]

        pad_value = 0.0
        stay_id = self.outcome_df.index.unique()[idx]

        # slice to make sure to always return a DF
        window = self.features_df.loc[stay_id:stay_id].to_numpy()
        labels = self.outcome_df.loc[stay_id:stay_id][self.vars["LABEL"]].to_numpy(dtype=np.float32)

        if len(labels) == 1:
            # only one label per stay, align with window
            labels = np.concatenate([np.empty(window.shape[0] - 1) * np.nan, labels], axis=0)

        length_diff = self.maxlen - window.shape[0]
        pad_mask = np.ones(window.shape[0])

        # Padding the array to fulfill size requirement
        if length_diff > 0:
            # window shorter than the longest window in dataset, pad to same length
            window = np.concatenate([window, np.ones((length_diff, window.shape[1])) * pad_value], axis=0)
            labels = np.concatenate([labels, np.ones(length_diff) * pad_value], axis=0)
            pad_mask = np.concatenate([pad_mask, np.zeros(length_diff)], axis=0)

        not_labeled = np.argwhere(np.isnan(labels))
        if len(not_labeled) > 0:
            labels[not_labeled] = -1
            pad_mask[not_labeled] = 0

        pad_mask = pad_mask.astype(bool)
        labels = labels.astype(np.float32)
        data = window.astype(np.float32)

        return from_numpy(data), from_numpy(labels), from_numpy(pad_mask)

    def get_balance(self) -> list:
        """Return the weight balance for the split of interest.

        Returns:
            Weights for each label.
        """
        counts = self.outcome_df[self.vars["LABEL"]].value_counts()
        return list((1 / counts) * np.sum(counts) / counts.shape[0])

    def get_data_and_labels(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Function to return all the data and labels aligned at once.

        We use this function for the ML methods which don't require an iterator.

        Returns:
            A Tuple containing data points and label for the split.
        """
        labels = self.outcome_df[self.vars["LABEL"]].to_numpy().astype(np.float32)
        rep = self.features_df
        if len(labels) == self.num_stays:
            rep = rep.groupby(level=self.vars["GROUP"], sort=False).last().reindex(self.outcome_df.index)
        row_indicators = self.outcome_df.index.to_numpy().reshape(-1, 1)
        rep = rep.to_numpy().astype(np.float32)
        valid = np.isfinite(labels.reshape(len(labels), -1)).all(axis=1)
        return rep[valid], labels[valid], row_indicators[valid]

    def to_tensor(self) -> tuple[Tensor, Tensor, Tensor]:
        return tuple(from_numpy(value) for value in self.get_data_and_labels())


@gin.configurable("GRUDPredictionPandasDataset")
class GRUDPredictionPandasDataset(PredictionPandasDataset):
    """Pandas equivalent of :class:`GRUDPredictionPolarsDataset`."""

    def ram_cache(self, cache: bool = True):
        self._cached_dataset = None
        if cache and hasattr(self, "_grud_dynamic_arrays"):
            logging.info(f"Caching {self.split} GRU-D dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __init__(
        self,
        *args,
        reference_means: Optional[np.ndarray] = None,
        time_unit: str = gin.REQUIRED,
        ram_cache: bool = True,
        **kwargs,
    ):
        data = args[0] if args else kwargs["data"]
        _grud_validate_time_unit(time_unit)
        super().__init__(*args, ram_cache=False, **kwargs)
        raw_features = data[self.split][DataSegment.features].sort_values(
            [self.vars["GROUP"], self.vars["SEQUENCE"]], kind="stable"
        )
        dynamic, indicators, static = _grud_validate_columns(raw_features, self.vars)
        self._grud_dynamic_names = dynamic
        self._grud_indicator_names = indicators
        self._grud_static_names = static
        self._grud_dynamic_arrays = {}
        self._grud_mask_arrays = {}
        self._grud_static_arrays = {}
        self._grud_sequence_arrays = {}
        self._grud_label_arrays = {}
        self._grud_stay_order = list(self.outcome_df.index.unique())
        self.time_unit = time_unit
        group = self.vars["GROUP"]
        label = self.vars["LABEL"]
        outcome = data[self.split][DataSegment.outcome]
        for stay_id, part in raw_features.groupby(group, sort=False):
            self._grud_dynamic_arrays[stay_id] = part[dynamic].to_numpy(dtype=np.float32)
            self._grud_mask_arrays[stay_id] = 1.0 - part[indicators].to_numpy(dtype=np.float32)
            self._grud_static_arrays[stay_id] = (
                part[static].iloc[0].to_numpy(dtype=np.float32) if static else np.zeros(0, dtype=np.float32)
            )
            self._grud_sequence_arrays[stay_id] = part[self.vars["SEQUENCE"]].to_numpy()
            self._grud_label_arrays[stay_id] = outcome.loc[outcome[group] == stay_id, label].to_numpy(dtype=np.float32)

        self.reference_means = (
            np.asarray(reference_means, dtype=np.float32).copy()
            if reference_means is not None
            else _grud_reference_means(data[DataSplit.train][DataSegment.features], self.vars)
        )
        if self.reference_means.shape != (len(dynamic),):
            raise ValueError(f"Expected {len(dynamic)} GRU-D reference means, got shape {self.reference_means.shape}.")
        self.ram_cache(ram_cache)

    def get_reference_means(self) -> np.ndarray:
        return self.reference_means.copy()

    def _build_item(self, idx: int) -> tuple[list[Tensor], Tensor, Tensor]:
        stay_id = self._grud_stay_order[idx]
        values = self._grud_dynamic_arrays[stay_id].copy()
        observed_mask = self._grud_mask_arrays[stay_id].copy()
        elapsed_time = _grud_elapsed_time(
            self._grud_sequence_arrays[stay_id], observed_mask, self.time_unit, stay_id
        )
        return _grud_pad_item(
            values,
            observed_mask,
            elapsed_time,
            self._grud_static_arrays[stay_id].copy(),
            self._grud_label_arrays[stay_id].copy(),
            self.maxlen,
        )

    def __getitem__(self, idx: int) -> tuple[list[Tensor], Tensor, Tensor]:
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        return self._build_item(idx)


@gin.configurable("ImputationPandasDataset")
class ImputationPandasDataset(CommonPandasDataset):
    """Subclass of Common Dataset that contains data for imputation models."""

    def __init__(
        self,
        data: dict[str, DataFrame],
        split: str = DataSplit.train,
        vars: dict[str, Union[str, list[str]]] | object = gin.REQUIRED,
        mask_proportion=0.3,
        mask_method="MCAR",
        mask_observation_proportion=0.3,
        ram_cache: bool = True,
    ):
        """
        Args:
            data (Dict[str, DataFrame]): data to use
            split (str, optional): split to apply. Defaults to Split.train.
            vars (Dict[str, str], optional): contains names of columns in the data. Defaults to gin.REQUIRED.
            mask_proportion (float, optional): proportion to artificially mask for amputation. Defaults to 0.3.
            mask_method (str, optional): masking mechanism. Defaults to "MCAR".
            mask_observation_proportion (float, optional): poportion of the observed data to be masked. Defaults to 0.3.
            ram_cache (bool, optional): if the dataset should be completely stored in ram and not generated on the fly during
                training. Defaults to True.
        """
        if not isinstance(vars, dict):
            raise ValueError(f"Expected vars to be of type dict, got {type(vars)} instead")
        super().__init__(data, split, vars, grouping_segment=DataSegment.static)
        self.amputated_values, self.amputation_mask = ampute_data(
            self.features_df, mask_method, mask_proportion, mask_observation_proportion
        )
        self.amputation_mask = (self.amputation_mask + self.features_df.isna().values).bool()
        self.amputation_mask = DataFrame(self.amputation_mask, columns=np.array(self.vars[DataSegment.dynamic]))
        self.amputation_mask[self.vars["GROUP"]] = self.features_df.index
        self.amputation_mask = self.amputation_mask.set_index(self.vars["GROUP"])

        self.target_missingness_mask = self.features_df.isna()
        self.features_df.fillna(0, inplace=True)
        self.ram_cache(ram_cache)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        stay_id = self.grouping_df.iloc[idx].name

        # slice to make sure to always return a DF
        window = self.features_df.loc[stay_id:stay_id, self.vars[DataSegment.dynamic]]
        window_missingness_mask = self.target_missingness_mask.loc[stay_id:stay_id, self.vars[DataSegment.dynamic]]
        amputated_window = self.amputated_values.loc[stay_id:stay_id, self.vars[DataSegment.dynamic]]
        amputation_mask = self.amputation_mask.loc[stay_id:stay_id, self.vars[DataSegment.dynamic]]

        return (
            from_numpy(amputated_window.values).to(float32),
            from_numpy(amputation_mask.values).to(float32),
            from_numpy(window.values).to(float32),
            from_numpy(window_missingness_mask.values).to(float32),
        )


@gin.configurable("ImputationPredictionDataset")
class ImputationPredictionDataset(Dataset):
    """Subclass of torch dataset that represents data with missingness for imputation.

    Args:
        data (DataFrame): dict of the different splits of the data
        grouping_column (str, optional): column that is used for grouping. Defaults to "stay_id".
        select_columns (List[str], optional): the columns to serve as input for the imputation model. Defaults to None.
        ram_cache (bool, optional): whether the dataset should be stored in ram. Defaults to True.
    """

    def __init__(
        self,
        data: DataFrame,
        grouping_column: Optional[str] = "stay_id",
        select_columns: Optional[list[str]] = None,
        ram_cache: bool = True,
    ):
        self.dyn_df = data

        if select_columns is not None:
            self.dyn_df = self.dyn_df[list(select_columns) + [grouping_column]]

        if grouping_column is not None:
            self.dyn_df = self.dyn_df.set_index(grouping_column)
        else:
            self.dyn_df = data

        # calculate basic info for the data
        self.group_indices = self.dyn_df.index.unique()
        self.maxlen = self.dyn_df.groupby(grouping_column).size().max()

        self._cached_dataset = None
        if ram_cache:
            logging.info("Caching dataset in ram.")
            self._cached_dataset = [self[i] for i in range(len(self))]

    def __len__(self) -> int:
        """Returns number of stays in the data.

        Returns:
            number of stays in the data
        """
        return self.group_indices.shape[0]

    def __getitem__(self, idx: int) -> Tensor:
        """Function to sample from the data split of choice.

        Used for deep learning implementations.

        Args:
            idx: A specific row index to sample.

        Returns:
            A sample from the data, consisting of data, labels and padding mask.
        """
        if self._cached_dataset is not None:
            return self._cached_dataset[idx]
        stay_id = self.group_indices[idx]

        # slice to make sure to always return a DF
        window = self.dyn_df.loc[stay_id:stay_id, :]

        return from_numpy(window.values).to(float32)
