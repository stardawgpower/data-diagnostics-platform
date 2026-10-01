from dataclasses import dataclass
from datetime import time, timedelta
from enum import StrEnum
from itertools import pairwise

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


class TemporalFrequency(StrEnum):
    """Supported calendar aggregation frequencies."""

    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"


class TemporalAggregation(StrEnum):
    """Supported numeric aggregation operations."""

    MEAN = "mean"
    MEDIAN = "median"
    SUM = "sum"
    MINIMUM = "min"
    MAXIMUM = "max"


@dataclass(frozen=True, slots=True)
class TemporalAxis:
    """Prepared timestamps aligned with the source dataset."""

    date_column: str
    time_column: str | None
    total_count: int
    valid_count: int
    invalid_count: int
    timestamps: pd.Series


@dataclass(frozen=True, slots=True)
class TemporalSamplingSummary:
    """Sampling and coverage diagnostics for a temporal axis."""

    total_count: int
    valid_count: int
    invalid_count: int
    unique_timestamp_count: int
    duplicate_row_count: int
    duplicate_timestamp_count: int
    is_chronologically_sorted: bool
    earliest: pd.Timestamp | None
    latest: pd.Timestamp | None
    span: pd.Timedelta | None
    interval_count: int
    minimum_interval: pd.Timedelta | None
    median_interval: pd.Timedelta | None
    maximum_interval: pd.Timedelta | None
    dominant_interval: pd.Timedelta | None
    dominant_interval_count: int
    dominant_interval_ratio: float | None


@dataclass(frozen=True, slots=True)
class TemporalGap:
    """One gap larger than an explicitly selected expected interval."""

    start: pd.Timestamp
    end: pd.Timestamp
    duration: pd.Timedelta
    expected_interval: pd.Timedelta
    excess_duration: pd.Timedelta


@dataclass(frozen=True, slots=True)
class TemporalGapSummary:
    """Gap diagnostics for one temporal axis."""

    expected_interval: pd.Timedelta
    unique_timestamp_count: int
    gap_count: int
    gaps: tuple[TemporalGap, ...]


@dataclass(frozen=True, slots=True)
class TemporalNumericSeries:
    """Finite numeric observations paired with valid timestamps."""

    value_column: str
    total_count: int
    valid_count: int
    excluded_count: int
    data: pd.DataFrame


@dataclass(frozen=True, slots=True)
class TemporalAggregationResult:
    """Calendar aggregation of a temporal numeric series."""

    value_column: str
    frequency: TemporalFrequency
    aggregation: TemporalAggregation
    total_count: int
    valid_count: int
    excluded_count: int
    bucket_count: int
    data: pd.DataFrame


_FREQUENCY_RULES = {
    TemporalFrequency.HOUR: "h",
    TemporalFrequency.DAY: "D",
    TemporalFrequency.WEEK: "W-MON",
    TemporalFrequency.MONTH: "MS",
}


def _require_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a dataset column or raise a clear validation error."""

    if column not in data.columns:
        raise ValueError(f"Unknown column: {column}.")

    return data[column]


def _require_numeric_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric, non-boolean dataset column."""

    series = _require_column(
        data,
        column,
    )

    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        raise ValueError(f"Column {column!r} must be numeric.")

    return series


def _parse_date_series(
    series: pd.Series,
    date_format: str | None,
) -> pd.Series:
    """
    Parse a date or datetime series without guessing textual formats.

    Native datetime columns can be used directly. Textual columns require
    an explicit format, normally supplied by schema inference or the user.
    """

    if is_datetime64_any_dtype(series.dtype):
        return pd.to_datetime(
            series,
            errors="coerce",
        )

    if date_format is None:
        raise ValueError("date_format is required for non-datetime date columns.")

    return pd.to_datetime(
        series,
        format=date_format,
        errors="coerce",
    )


def _python_time_offsets(
    series: pd.Series,
) -> pd.Series:
    """Convert Python time objects to timedeltas."""

    offsets: list[pd.Timedelta | pd.NaT] = []

    for value in series:
        if pd.isna(value):
            offsets.append(pd.NaT)
            continue

        offsets.append(
            pd.Timedelta(
                hours=value.hour,
                minutes=value.minute,
                seconds=value.second,
                microseconds=value.microsecond,
            )
        )

    return pd.Series(
        offsets,
        index=series.index,
        dtype="timedelta64[ns]",
    )


def _parse_time_offsets(
    series: pd.Series,
    time_format: str | None,
) -> pd.Series:
    """Parse time-of-day values into offsets from midnight."""

    if is_datetime64_any_dtype(series.dtype):
        parsed = pd.to_datetime(
            series,
            errors="coerce",
        )

    else:
        non_null = series.dropna()

        if not non_null.empty and bool(
            non_null.map(
                lambda value: isinstance(
                    value,
                    time,
                )
            ).all()
        ):
            return _python_time_offsets(series)

        if time_format is None:
            raise ValueError("time_format is required for non-datetime time columns.")

        parsed = pd.to_datetime(
            series,
            format=time_format,
            errors="coerce",
        )

    offsets = (
        pd.to_timedelta(
            parsed.dt.hour,
            unit="h",
        )
        + pd.to_timedelta(
            parsed.dt.minute,
            unit="m",
        )
        + pd.to_timedelta(
            parsed.dt.second,
            unit="s",
        )
        + pd.to_timedelta(
            parsed.dt.microsecond,
            unit="us",
        )
    )

    return offsets.where(parsed.notna())


def prepare_temporal_axis(
    data: pd.DataFrame,
    date_column: str,
    *,
    date_format: str | None = None,
    time_column: str | None = None,
    time_format: str | None = None,
) -> TemporalAxis:
    """
    Prepare timestamps for temporal analysis without modifying the dataset.

    A native datetime column can be used directly. Textual dates require an
    explicit date format. A separate time-of-day column may optionally be
    combined with the date column using an explicit time format.
    """

    date_series = _require_column(
        data,
        date_column,
    )

    if time_column is not None and time_column == date_column:
        raise ValueError("Date and time columns must be different.")

    parsed_dates = _parse_date_series(
        date_series,
        date_format,
    )

    if time_column is None:
        timestamps = parsed_dates.copy()

    else:
        time_series = _require_column(
            data,
            time_column,
        )

        time_offsets = _parse_time_offsets(
            time_series,
            time_format,
        )

        timestamps = parsed_dates.dt.normalize() + time_offsets

    timestamps = pd.Series(
        timestamps,
        index=data.index,
        name="timestamp",
    ).copy()

    valid_count = int(timestamps.notna().sum())

    return TemporalAxis(
        date_column=date_column,
        time_column=time_column,
        total_count=len(data),
        valid_count=valid_count,
        invalid_count=(len(data) - valid_count),
        timestamps=timestamps,
    )


def _sorted_unique_timestamps(
    temporal: TemporalAxis,
) -> pd.DatetimeIndex:
    """Return sorted unique valid timestamps."""

    valid = temporal.timestamps.dropna()

    if valid.empty:
        return pd.DatetimeIndex([])

    return pd.DatetimeIndex(valid.unique()).sort_values()


def analyze_temporal_sampling(
    temporal: TemporalAxis,
) -> TemporalSamplingSummary:
    """
    Describe coverage, duplicates, ordering, and sampling intervals.

    Interval diagnostics use sorted unique timestamps so duplicate rows do
    not create artificial zero-length sampling intervals.
    """

    valid = temporal.timestamps.dropna()

    if valid.empty:
        return TemporalSamplingSummary(
            total_count=temporal.total_count,
            valid_count=0,
            invalid_count=temporal.invalid_count,
            unique_timestamp_count=0,
            duplicate_row_count=0,
            duplicate_timestamp_count=0,
            is_chronologically_sorted=True,
            earliest=None,
            latest=None,
            span=None,
            interval_count=0,
            minimum_interval=None,
            median_interval=None,
            maximum_interval=None,
            dominant_interval=None,
            dominant_interval_count=0,
            dominant_interval_ratio=None,
        )

    timestamp_counts = valid.value_counts()

    duplicate_row_count = int(valid.duplicated(keep="first").sum())

    duplicate_timestamp_count = int((timestamp_counts > 1).sum())

    earliest = pd.Timestamp(valid.min())

    latest = pd.Timestamp(valid.max())

    unique_timestamps = _sorted_unique_timestamps(temporal)

    if len(unique_timestamps) < 2:
        return TemporalSamplingSummary(
            total_count=temporal.total_count,
            valid_count=len(valid),
            invalid_count=temporal.invalid_count,
            unique_timestamp_count=len(unique_timestamps),
            duplicate_row_count=duplicate_row_count,
            duplicate_timestamp_count=(duplicate_timestamp_count),
            is_chronologically_sorted=bool(valid.is_monotonic_increasing),
            earliest=earliest,
            latest=latest,
            span=latest - earliest,
            interval_count=0,
            minimum_interval=None,
            median_interval=None,
            maximum_interval=None,
            dominant_interval=None,
            dominant_interval_count=0,
            dominant_interval_ratio=None,
        )

    intervals = pd.Series(unique_timestamps[1:] - unique_timestamps[:-1])

    interval_frequency = intervals.value_counts()

    maximum_frequency = int(interval_frequency.max())

    modes = interval_frequency[interval_frequency == maximum_frequency]

    if len(modes) == 1:
        dominant_interval = pd.Timedelta(modes.index[0])

        dominant_interval_count = maximum_frequency

        dominant_interval_ratio = maximum_frequency / len(intervals)

    else:
        dominant_interval = None
        dominant_interval_count = 0
        dominant_interval_ratio = None

    return TemporalSamplingSummary(
        total_count=temporal.total_count,
        valid_count=len(valid),
        invalid_count=temporal.invalid_count,
        unique_timestamp_count=len(unique_timestamps),
        duplicate_row_count=duplicate_row_count,
        duplicate_timestamp_count=(duplicate_timestamp_count),
        is_chronologically_sorted=bool(valid.is_monotonic_increasing),
        earliest=earliest,
        latest=latest,
        span=latest - earliest,
        interval_count=len(intervals),
        minimum_interval=pd.Timedelta(intervals.min()),
        median_interval=pd.Timedelta(intervals.median()),
        maximum_interval=pd.Timedelta(intervals.max()),
        dominant_interval=dominant_interval,
        dominant_interval_count=(dominant_interval_count),
        dominant_interval_ratio=(dominant_interval_ratio),
    )


def _coerce_expected_interval(
    expected_interval: (str | pd.Timedelta | timedelta),
) -> pd.Timedelta:
    """Normalize and validate an explicitly supplied expected interval."""

    if isinstance(
        expected_interval,
        (bool, int, float),
    ):
        raise TypeError("expected_interval must be a timedelta or duration string.")

    try:
        interval = pd.Timedelta(expected_interval)
    except (
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError("expected_interval is not a valid duration.") from exc

    if pd.isna(interval) or interval <= pd.Timedelta(0):
        raise ValueError("expected_interval must be greater than zero.")

    return interval


def detect_time_gaps(
    temporal: TemporalAxis,
    expected_interval: (str | pd.Timedelta | timedelta),
) -> TemporalGapSummary:
    """
    Detect intervals larger than an explicitly selected expected interval.

    No missing rows are created and no interpolation is performed.
    """

    expected = _coerce_expected_interval(expected_interval)

    unique_timestamps = _sorted_unique_timestamps(temporal)

    gaps: list[TemporalGap] = []

    for start, end in pairwise(
        unique_timestamps,
    ):
        start_timestamp = pd.Timestamp(start)

        end_timestamp = pd.Timestamp(end)

        duration = pd.Timedelta(end_timestamp - start_timestamp)

        if duration <= expected:
            continue

        gaps.append(
            TemporalGap(
                start=start_timestamp,
                end=end_timestamp,
                duration=duration,
                expected_interval=expected,
                excess_duration=(duration - expected),
            )
        )

    return TemporalGapSummary(
        expected_interval=expected,
        unique_timestamp_count=len(unique_timestamps),
        gap_count=len(gaps),
        gaps=tuple(gaps),
    )


def prepare_temporal_numeric_series(
    data: pd.DataFrame,
    temporal: TemporalAxis,
    value_column: str,
) -> TemporalNumericSeries:
    """
    Prepare finite numeric observations in chronological order.

    Sorting applies only to the returned analysis frame. The source
    DataFrame and working dataset order are never changed.
    """

    if len(temporal.timestamps) != len(data) or not temporal.timestamps.index.equals(data.index):
        raise ValueError("Temporal timestamps must align with the source dataset.")

    series = _require_numeric_column(
        data,
        value_column,
    )

    numeric_values = series.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    frame = pd.DataFrame(
        {
            "timestamp": (temporal.timestamps.copy()),
            "value": numeric_values,
        },
        index=data.index,
    )

    valid_mask = frame["timestamp"].notna() & np.isfinite(frame["value"])

    prepared = (
        frame.loc[
            valid_mask,
            [
                "timestamp",
                "value",
            ],
        ]
        .sort_values(
            by="timestamp",
            kind="mergesort",
        )
        .copy()
    )

    valid_count = len(prepared)

    return TemporalNumericSeries(
        value_column=value_column,
        total_count=len(data),
        valid_count=valid_count,
        excluded_count=(len(data) - valid_count),
        data=prepared,
    )


def _coerce_frequency(
    frequency: TemporalFrequency | str,
) -> TemporalFrequency:
    """Normalize a temporal aggregation frequency."""

    if isinstance(
        frequency,
        TemporalFrequency,
    ):
        return frequency

    if not isinstance(
        frequency,
        str,
    ):
        raise TypeError("frequency must be a string or TemporalFrequency.")

    try:
        return TemporalFrequency(frequency.lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported temporal frequency: {frequency!r}.") from exc


def _coerce_aggregation(
    aggregation: (TemporalAggregation | str),
) -> TemporalAggregation:
    """Normalize a temporal aggregation operation."""

    if isinstance(
        aggregation,
        TemporalAggregation,
    ):
        return aggregation

    if not isinstance(
        aggregation,
        str,
    ):
        raise TypeError("aggregation must be a string or TemporalAggregation.")

    try:
        return TemporalAggregation(aggregation.lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported temporal aggregation: {aggregation!r}.") from exc


def aggregate_temporal_numeric_series(
    data: pd.DataFrame,
    temporal: TemporalAxis,
    value_column: str,
    *,
    frequency: (TemporalFrequency | str) = TemporalFrequency.DAY,
    aggregation: (TemporalAggregation | str) = TemporalAggregation.MEAN,
) -> TemporalAggregationResult:
    """
    Aggregate finite numeric observations into calendar buckets.

    Empty buckets are omitted rather than being filled or interpolated.
    Weeks are represented as Monday-starting calendar buckets.
    """

    normalized_frequency = _coerce_frequency(frequency)

    normalized_aggregation = _coerce_aggregation(aggregation)

    temporal_series = prepare_temporal_numeric_series(
        data,
        temporal,
        value_column,
    )

    if temporal_series.data.empty:
        empty = pd.DataFrame(
            columns=[
                "timestamp",
                "value",
                "count",
            ]
        )

        return TemporalAggregationResult(
            value_column=value_column,
            frequency=normalized_frequency,
            aggregation=normalized_aggregation,
            total_count=(temporal_series.total_count),
            valid_count=0,
            excluded_count=(temporal_series.excluded_count),
            bucket_count=0,
            data=empty,
        )

    rule = _FREQUENCY_RULES[normalized_frequency]

    resampler = temporal_series.data.resample(
        rule,
        on="timestamp",
        label="left",
        closed="left",
    )["value"]

    if normalized_aggregation == TemporalAggregation.MEAN:
        values = resampler.mean()

    elif normalized_aggregation == TemporalAggregation.MEDIAN:
        values = resampler.median()

    elif normalized_aggregation == TemporalAggregation.SUM:
        values = resampler.sum(min_count=1)

    elif normalized_aggregation == TemporalAggregation.MINIMUM:
        values = resampler.min()

    else:
        values = resampler.max()

    counts = resampler.count().astype(int)

    aggregated = pd.DataFrame(
        {
            "value": values,
            "count": counts,
        }
    )

    aggregated = aggregated.loc[aggregated["count"] > 0].reset_index().copy()

    return TemporalAggregationResult(
        value_column=value_column,
        frequency=normalized_frequency,
        aggregation=normalized_aggregation,
        total_count=(temporal_series.total_count),
        valid_count=(temporal_series.valid_count),
        excluded_count=(temporal_series.excluded_count),
        bucket_count=len(aggregated),
        data=aggregated,
    )
