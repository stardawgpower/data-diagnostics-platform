from dataclasses import dataclass

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


@dataclass(frozen=True, slots=True)
class ColumnMissingness:
    """Missing-value summary for one column."""

    column: str
    missing_count: int
    missing_ratio: float


@dataclass(frozen=True, slots=True)
class HistogramBin:
    """One numeric histogram interval."""

    left: float
    right: float
    count: int


@dataclass(frozen=True, slots=True)
class NumericDistribution:
    """Exploratory distribution summary for one numeric column."""

    column: str
    total_count: int
    valid_count: int
    missing_count: int
    non_finite_count: int
    mean: float | None
    standard_deviation: float | None
    minimum: float | None
    first_quartile: float | None
    median: float | None
    third_quartile: float | None
    maximum: float | None
    histogram: tuple[HistogramBin, ...]


@dataclass(frozen=True, slots=True)
class NumericOutlierSummary:
    """IQR-based exploratory outlier summary for one numeric column."""

    column: str
    total_count: int
    valid_count: int
    first_quartile: float | None
    third_quartile: float | None
    iqr: float | None
    lower_bound: float | None
    upper_bound: float | None
    lower_outlier_count: int
    upper_outlier_count: int
    outlier_count: int
    outlier_ratio: float | None


@dataclass(frozen=True, slots=True)
class CategoryFrequency:
    """Frequency of one observed category."""

    value: object
    count: int
    ratio: float


@dataclass(frozen=True, slots=True)
class CategoricalFrequencySummary:
    """Top-frequency summary for one column."""

    column: str
    total_count: int
    non_missing_count: int
    missing_count: int
    unique_count: int
    frequencies: tuple[CategoryFrequency, ...]
    other_count: int
    other_ratio: float


@dataclass(frozen=True, slots=True)
class NumericPairData:
    """Finite paired numeric observations prepared for exploration."""

    x_column: str
    y_column: str
    total_count: int
    valid_count: int
    excluded_count: int
    data: pd.DataFrame


def _require_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a column or raise a clear error if it does not exist."""

    if column not in data.columns:
        raise ValueError(f"Unknown column: {column}.")

    return data[column]


def _require_numeric_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric non-boolean column."""

    series = _require_column(
        data,
        column,
    )

    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        raise ValueError(f"Column {column!r} must be numeric.")

    return series


def _numeric_array(
    data: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Return one numeric column as a floating-point array."""

    series = _require_numeric_column(
        data,
        column,
    )

    return series.to_numpy(
        dtype=float,
        na_value=np.nan,
    )


def _finite_numeric_values(
    data: pd.DataFrame,
    column: str,
) -> tuple[pd.Series, int, int]:
    """Return finite values with missing and non-finite counts."""

    series = _require_numeric_column(
        data,
        column,
    )

    values = series.to_numpy(
        dtype=float,
        na_value=np.nan,
    )

    finite_mask = np.isfinite(values)

    finite_values = pd.Series(
        values[finite_mask],
        dtype="float64",
    )

    missing_count = int(series.isna().sum())

    non_finite_count = int(np.isinf(values).sum())

    return (
        finite_values,
        missing_count,
        non_finite_count,
    )


def _optional_float(
    value: object,
) -> float | None:
    """Convert a scalar to float unless it is missing."""

    if pd.isna(value):
        return None

    return float(value)


def summarize_missingness(
    data: pd.DataFrame,
) -> tuple[ColumnMissingness, ...]:
    """Summarize missing values for every column without modifying the dataset."""

    row_count = len(data)

    summaries = []

    for column in data.columns:
        missing_count = int(data[column].isna().sum())

        missing_ratio = missing_count / row_count if row_count else 0.0

        summaries.append(
            ColumnMissingness(
                column=str(column),
                missing_count=missing_count,
                missing_ratio=missing_ratio,
            )
        )

    return tuple(summaries)


def summarize_numeric_distribution(
    data: pd.DataFrame,
    column: str,
    *,
    bins: int = 20,
) -> NumericDistribution:
    """
    Build a distribution summary for one numeric column.

    Missing values and positive/negative infinity are excluded from
    descriptive statistics and histogram construction.
    """

    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("Histogram bins must be a positive integer.")

    values, missing_count, non_finite_count = _finite_numeric_values(
        data,
        column,
    )

    if values.empty:
        return NumericDistribution(
            column=column,
            total_count=len(data),
            valid_count=0,
            missing_count=missing_count,
            non_finite_count=non_finite_count,
            mean=None,
            standard_deviation=None,
            minimum=None,
            first_quartile=None,
            median=None,
            third_quartile=None,
            maximum=None,
            histogram=(),
        )

    counts, edges = np.histogram(
        values.to_numpy(),
        bins=bins,
    )

    histogram = tuple(
        HistogramBin(
            left=float(edges[index]),
            right=float(edges[index + 1]),
            count=int(count),
        )
        for index, count in enumerate(counts)
    )

    return NumericDistribution(
        column=column,
        total_count=len(data),
        valid_count=len(values),
        missing_count=missing_count,
        non_finite_count=non_finite_count,
        mean=_optional_float(values.mean()),
        standard_deviation=_optional_float(values.std()),
        minimum=_optional_float(values.min()),
        first_quartile=_optional_float(values.quantile(0.25)),
        median=_optional_float(values.median()),
        third_quartile=_optional_float(values.quantile(0.75)),
        maximum=_optional_float(values.max()),
        histogram=histogram,
    )


def summarize_numeric_outliers(
    data: pd.DataFrame,
    column: str,
    *,
    multiplier: float = 1.5,
) -> NumericOutlierSummary:
    """
    Describe potential outliers using the interquartile-range rule.

    This is a descriptive diagnostic only. Outliers are not treated as
    erroneous data and no values are removed or modified.
    """

    if isinstance(multiplier, bool):
        raise TypeError("IQR multiplier must be a positive finite number.")

    try:
        multiplier_value = float(multiplier)
    except (TypeError, ValueError) as exc:
        raise ValueError("IQR multiplier must be a positive finite number.") from exc

    if not np.isfinite(multiplier_value) or multiplier_value <= 0:
        raise ValueError("IQR multiplier must be a positive finite number.")

    values, _, _ = _finite_numeric_values(
        data,
        column,
    )

    if values.empty:
        return NumericOutlierSummary(
            column=column,
            total_count=len(data),
            valid_count=0,
            first_quartile=None,
            third_quartile=None,
            iqr=None,
            lower_bound=None,
            upper_bound=None,
            lower_outlier_count=0,
            upper_outlier_count=0,
            outlier_count=0,
            outlier_ratio=None,
        )

    first_quartile = float(values.quantile(0.25))
    third_quartile = float(values.quantile(0.75))

    iqr = third_quartile - first_quartile

    lower_bound = first_quartile - (multiplier_value * iqr)

    upper_bound = third_quartile + (multiplier_value * iqr)

    lower_outlier_count = int((values < lower_bound).sum())

    upper_outlier_count = int((values > upper_bound).sum())

    outlier_count = lower_outlier_count + upper_outlier_count

    return NumericOutlierSummary(
        column=column,
        total_count=len(data),
        valid_count=len(values),
        first_quartile=first_quartile,
        third_quartile=third_quartile,
        iqr=iqr,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        lower_outlier_count=lower_outlier_count,
        upper_outlier_count=upper_outlier_count,
        outlier_count=outlier_count,
        outlier_ratio=(outlier_count / len(values)),
    )


def summarize_categorical_frequencies(
    data: pd.DataFrame,
    column: str,
    *,
    top_n: int = 10,
) -> CategoricalFrequencySummary:
    """Build a top-N categorical frequency summary."""

    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("top_n must be a positive integer.")

    series = _require_column(
        data,
        column,
    )

    total_count = len(series)
    missing_count = int(series.isna().sum())

    non_missing = series.dropna()

    non_missing_count = len(non_missing)

    value_counts = non_missing.value_counts(
        dropna=True,
    )

    top_counts = value_counts.head(top_n)

    denominator = total_count

    frequencies = tuple(
        CategoryFrequency(
            value=value,
            count=int(count),
            ratio=(int(count) / denominator if denominator else 0.0),
        )
        for value, count in top_counts.items()
    )

    displayed_count = int(top_counts.sum())

    other_count = non_missing_count - displayed_count

    other_ratio = other_count / denominator if denominator else 0.0

    return CategoricalFrequencySummary(
        column=column,
        total_count=total_count,
        non_missing_count=non_missing_count,
        missing_count=missing_count,
        unique_count=int(
            non_missing.nunique(
                dropna=True,
            )
        ),
        frequencies=frequencies,
        other_count=other_count,
        other_ratio=other_ratio,
    )


def prepare_numeric_pair(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> NumericPairData:
    """
    Prepare paired finite numeric observations for visual exploration.

    This function does not calculate correlation or fit any statistical model.
    """

    if x_column == y_column:
        raise ValueError("Numeric pair columns must be different.")

    x_values = _numeric_array(
        data,
        x_column,
    )

    y_values = _numeric_array(
        data,
        y_column,
    )

    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)

    pair_data = pd.DataFrame(
        {
            x_column: x_values[valid_mask],
            y_column: y_values[valid_mask],
        },
        index=data.index[valid_mask],
    ).copy()

    valid_count = len(pair_data)

    return NumericPairData(
        x_column=x_column,
        y_column=y_column,
        total_count=len(data),
        valid_count=valid_count,
        excluded_count=(len(data) - valid_count),
        data=pair_data,
    )
