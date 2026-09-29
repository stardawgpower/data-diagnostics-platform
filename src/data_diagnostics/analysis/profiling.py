from dataclasses import dataclass

import pandas as pd

from data_diagnostics.ingestion import (
    ColumnSchema,
    DatasetSchema,
    SemanticType,
    infer_schema,
)
from data_diagnostics.quality import (
    DataQualityReport,
    analyze_data_quality,
)


@dataclass(frozen=True, slots=True)
class NumericSummary:
    """Descriptive statistics for a numeric column."""

    count: int
    mean: float | None
    median: float | None
    standard_deviation: float | None
    minimum: float | None
    maximum: float | None


@dataclass(frozen=True, slots=True)
class CategoricalSummary:
    """Descriptive statistics for a categorical or text-like column."""

    count: int
    unique_count: int
    most_frequent_value: object | None
    most_frequent_count: int | None


@dataclass(frozen=True, slots=True)
class DatetimeSummary:
    """Descriptive statistics for a datetime column."""

    count: int
    earliest: pd.Timestamp | None
    latest: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Combined schema and descriptive information for one column."""

    schema: ColumnSchema
    numeric: NumericSummary | None = None
    categorical: CategoricalSummary | None = None
    datetime: DatetimeSummary | None = None


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Complete profile of a dataset."""

    row_count: int
    column_count: int
    schema: DatasetSchema
    quality: DataQualityReport
    columns: tuple[ColumnProfile, ...]


def _optional_float(value: object) -> float | None:
    """Convert a numeric scalar to float while representing missing values as None."""

    if pd.isna(value):
        return None

    return float(value)


def _build_numeric_summary(series: pd.Series) -> NumericSummary:
    """Build descriptive statistics for a numeric series."""

    non_null = series.dropna()

    return NumericSummary(
        count=int(non_null.count()),
        mean=_optional_float(non_null.mean()),
        median=_optional_float(non_null.median()),
        standard_deviation=_optional_float(non_null.std()),
        minimum=_optional_float(non_null.min()),
        maximum=_optional_float(non_null.max()),
    )


def _build_categorical_summary(series: pd.Series) -> CategoricalSummary:
    """Build descriptive statistics for a categorical or text-like series."""

    non_null = series.dropna()
    value_counts = non_null.value_counts(dropna=True)

    if value_counts.empty:
        most_frequent_value = None
        most_frequent_count = None
    else:
        most_frequent_value = value_counts.index[0]
        most_frequent_count = int(value_counts.iloc[0])

    return CategoricalSummary(
        count=int(non_null.count()),
        unique_count=int(non_null.nunique(dropna=True)),
        most_frequent_value=most_frequent_value,
        most_frequent_count=most_frequent_count,
    )


def _build_datetime_summary(series: pd.Series) -> DatetimeSummary:
    """Build descriptive statistics for a datetime-like series."""

    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        parsed = series.dropna()
    else:
        parsed = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        ).dropna()

    if parsed.empty:
        earliest = None
        latest = None
    else:
        earliest = pd.Timestamp(parsed.min())
        latest = pd.Timestamp(parsed.max())

    return DatetimeSummary(
        count=int(parsed.count()),
        earliest=earliest,
        latest=latest,
    )


def _profile_column(
    data: pd.DataFrame,
    column_schema: ColumnSchema,
) -> ColumnProfile:
    """Build a profile for a single dataset column."""

    series = data[column_schema.name]
    semantic_type = column_schema.semantic_type

    if semantic_type in {
        SemanticType.NUMERIC_DISCRETE,
        SemanticType.NUMERIC_CONTINUOUS,
    }:
        return ColumnProfile(
            schema=column_schema,
            numeric=_build_numeric_summary(series),
        )

    if semantic_type == SemanticType.DATETIME:
        return ColumnProfile(
            schema=column_schema,
            datetime=_build_datetime_summary(series),
        )

    if semantic_type in {
        SemanticType.IDENTIFIER,
        SemanticType.BINARY,
        SemanticType.CATEGORICAL,
        SemanticType.TEXT,
    }:
        return ColumnProfile(
            schema=column_schema,
            categorical=_build_categorical_summary(series),
        )

    return ColumnProfile(schema=column_schema)


def profile_dataset(data: pd.DataFrame) -> DatasetProfile:
    """
    Build a structured profile for an entire dataset.

    Profiling combines schema inference, data-quality diagnostics,
    and descriptive statistics without modifying the input DataFrame.
    """

    schema = infer_schema(data)
    quality = analyze_data_quality(data)

    columns = tuple(_profile_column(data, column_schema) for column_schema in schema.columns)

    return DatasetProfile(
        row_count=len(data),
        column_count=len(data.columns),
        schema=schema,
        quality=quality,
        columns=columns,
    )
