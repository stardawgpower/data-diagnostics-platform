import re
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


class SemanticType(str, Enum):
    """Semantic meaning inferred for a dataset column."""

    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    TIME = "time"
    BINARY = "binary"
    CATEGORICAL = "categorical"
    NUMERIC_DISCRETE = "numeric_discrete"
    NUMERIC_CONTINUOUS = "numeric_continuous"
    TEXT = "text"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ColumnSchema:
    """Metadata inferred for one dataset column."""

    name: str
    physical_dtype: str
    semantic_type: SemanticType
    missing_count: int
    missing_ratio: float
    unique_count: int
    unique_ratio: float
    confidence: float
    reason: str
    temporal_format: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSchema:
    """Schema metadata inferred for an entire dataset."""

    row_count: int
    column_count: int
    columns: tuple[ColumnSchema, ...]


_IDENTIFIER_PATTERN = re.compile(
    r"(^id$|_id$|^id_|uuid|guid|_key$|^key$)",
    re.IGNORECASE,
)

_DATETIME_FORMATS = (
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)

_TIME_FORMATS = (
    "%H:%M:%S",
    "%H.%M.%S",
    "%H:%M",
    "%H.%M",
)


def _looks_like_identifier(name: str, unique_ratio: float) -> bool:
    """Return whether a column name and uniqueness suggest an identifier."""

    return bool(_IDENTIFIER_PATTERN.search(name)) and unique_ratio >= 0.90


def _best_temporal_format(
    series: pd.Series,
    formats: tuple[str, ...],
) -> tuple[str | None, float, bool]:
    """
    Find the strict temporal format that parses the largest share of values.

    Returns
    -------
    tuple
        Best format, parse ratio, and whether the result is ambiguous.
    """

    non_null = series.dropna()

    if non_null.empty:
        return None, 0.0, False

    sample = non_null.astype("string").head(500)
    scores: list[tuple[str, float]] = []

    for format_string in formats:
        parsed = pd.to_datetime(
            sample,
            format=format_string,
            errors="coerce",
        )
        scores.append((format_string, float(parsed.notna().mean())))

    best_ratio = max(score for _, score in scores)

    if best_ratio == 0:
        return None, 0.0, False

    best_formats = [format_string for format_string, score in scores if score == best_ratio]

    if len(best_formats) > 1:
        return None, best_ratio, True

    return best_formats[0], best_ratio, False


def _infer_temporal_string(
    series: pd.Series,
) -> tuple[SemanticType | None, float, str, str | None]:
    """Infer a strict date/datetime or time-only representation."""

    time_format, time_ratio, time_ambiguous = _best_temporal_format(
        series,
        _TIME_FORMATS,
    )

    if time_ratio >= 0.80:
        if time_ambiguous:
            return (
                SemanticType.TIME,
                0.60,
                "Values appear to represent times, but the format is ambiguous.",
                None,
            )

        return (
            SemanticType.TIME,
            time_ratio,
            f"String values match time format {time_format}.",
            time_format,
        )

    datetime_format, datetime_ratio, datetime_ambiguous = _best_temporal_format(
        series,
        _DATETIME_FORMATS,
    )

    if datetime_ratio >= 0.80:
        if datetime_ambiguous:
            return (
                SemanticType.DATETIME,
                0.60,
                (
                    "Values appear to represent dates, but day/month order "
                    "is ambiguous and requires confirmation."
                ),
                None,
            )

        return (
            SemanticType.DATETIME,
            datetime_ratio,
            f"String values match datetime format {datetime_format}.",
            datetime_format,
        )

    return None, 0.0, "", None


def _is_integer_like_numeric(series: pd.Series) -> bool:
    """Return whether every non-null numeric value is integer-like."""

    non_null = pd.to_numeric(series, errors="coerce").dropna()

    if non_null.empty:
        return False

    return bool(((non_null % 1) == 0).all())


def _infer_numeric_semantic_type(
    series: pd.Series,
    unique_count: int,
    unique_ratio: float,
) -> tuple[SemanticType, float, str, str | None]:
    """Infer whether a numeric column is better treated as discrete or continuous."""

    if not _is_integer_like_numeric(series):
        return (
            SemanticType.NUMERIC_CONTINUOUS,
            0.90,
            "Numeric column contains fractional values.",
            None,
        )

    low_cardinality = unique_count <= 20 or (unique_count <= 50 and unique_ratio <= 0.20)

    if low_cardinality:
        return (
            SemanticType.NUMERIC_DISCRETE,
            0.85,
            "Integer-like numeric column has relatively low cardinality.",
            None,
        )

    return (
        SemanticType.NUMERIC_CONTINUOUS,
        0.80,
        (
            "Integer-like numeric column has high cardinality and is treated "
            "as continuous for analysis."
        ),
        None,
    )


def _infer_semantic_type(
    series: pd.Series,
    name: str,
    unique_count: int,
    unique_ratio: float,
) -> tuple[SemanticType, float, str, str | None]:
    """Infer a semantic type for a single column."""

    non_null = series.dropna()

    if non_null.empty:
        return (
            SemanticType.UNKNOWN,
            1.0,
            "Column contains only missing values.",
            None,
        )

    if is_datetime64_any_dtype(series.dtype):
        return (
            SemanticType.DATETIME,
            1.0,
            "Physical dtype is datetime.",
            None,
        )

    if is_bool_dtype(series.dtype):
        return (
            SemanticType.BINARY,
            1.0,
            "Physical dtype is boolean.",
            None,
        )

    if _looks_like_identifier(name, unique_ratio):
        return (
            SemanticType.IDENTIFIER,
            0.95,
            "Column name suggests an identifier and values are highly unique.",
            None,
        )

    is_textual = (
        is_string_dtype(series.dtype)
        or is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    if is_textual:
        temporal_type, confidence, reason, temporal_format = _infer_temporal_string(series)

        if temporal_type is not None:
            return temporal_type, confidence, reason, temporal_format

    if unique_count == 2:
        return (
            SemanticType.BINARY,
            0.95,
            "Column contains exactly two distinct non-null values.",
            None,
        )

    if is_numeric_dtype(series.dtype):
        return _infer_numeric_semantic_type(
            series,
            unique_count,
            unique_ratio,
        )

    if is_textual:
        if isinstance(series.dtype, pd.CategoricalDtype):
            return (
                SemanticType.CATEGORICAL,
                0.95,
                "Physical dtype is categorical.",
                None,
            )

        low_cardinality = (unique_count <= 20 and unique_ratio < 0.90) or (
            unique_count <= 50 and unique_ratio <= 0.20
        )

        if low_cardinality:
            return (
                SemanticType.CATEGORICAL,
                0.85,
                "String column has relatively low cardinality.",
                None,
            )

        return (
            SemanticType.TEXT,
            0.75,
            "String column has relatively high cardinality.",
            None,
        )

    return (
        SemanticType.UNKNOWN,
        0.50,
        "No supported semantic-type rule matched the column.",
        None,
    )


def infer_schema(data: pd.DataFrame) -> DatasetSchema:
    """
    Infer semantic metadata for every column in a DataFrame.

    The input DataFrame is never modified.
    """

    row_count = len(data)
    column_schemas: list[ColumnSchema] = []

    for column_name in data.columns:
        series = data[column_name]

        missing_count = int(series.isna().sum())
        non_null_count = int(series.notna().sum())
        unique_count = int(series.nunique(dropna=True))

        missing_ratio = missing_count / row_count if row_count else 0.0
        unique_ratio = unique_count / non_null_count if non_null_count else 0.0

        semantic_type, confidence, reason, temporal_format = _infer_semantic_type(
            series=series,
            name=str(column_name),
            unique_count=unique_count,
            unique_ratio=unique_ratio,
        )

        column_schemas.append(
            ColumnSchema(
                name=str(column_name),
                physical_dtype=str(series.dtype),
                semantic_type=semantic_type,
                missing_count=missing_count,
                missing_ratio=missing_ratio,
                unique_count=unique_count,
                unique_ratio=unique_ratio,
                confidence=confidence,
                reason=reason,
                temporal_format=temporal_format,
            )
        )

    return DatasetSchema(
        row_count=row_count,
        column_count=len(data.columns),
        columns=tuple(column_schemas),
    )
