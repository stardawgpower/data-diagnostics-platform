import re
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


class SemanticType(str, Enum):
    """Semantic meaning inferred for a dataset column."""

    IDENTIFIER = "identifier"
    DATETIME = "datetime"
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

_DATETIME_NAME_PATTERN = re.compile(
    r"(date|time|timestamp|created|updated|datetime|dob)",
    re.IGNORECASE,
)


def _looks_like_identifier(name: str, unique_ratio: float) -> bool:
    """Return whether a column name and uniqueness suggest an identifier."""

    return bool(_IDENTIFIER_PATTERN.search(name)) and unique_ratio >= 0.90


def _looks_like_datetime_values(series: pd.Series, name: str) -> bool:
    """Return whether string values are plausible datetime candidates."""

    non_null = series.dropna()

    if non_null.empty:
        return False

    if _DATETIME_NAME_PATTERN.search(name):
        return True

    sample = non_null.astype("string").head(200)

    date_like_ratio = sample.str.contains(
        r"[-/:T]",
        regex=True,
        na=False,
    ).mean()

    return bool(date_like_ratio >= 0.50)


def _datetime_parse_ratio(series: pd.Series) -> float:
    """Calculate the proportion of non-null values parseable as datetimes."""

    non_null = series.dropna()

    if non_null.empty:
        return 0.0

    sample = non_null.astype("string").head(200)

    parsed = pd.to_datetime(
        sample,
        errors="coerce",
        format="mixed",
    )

    return float(parsed.notna().mean())


def _is_integer_like_float(series: pd.Series) -> bool:
    """Return whether every non-null floating-point value is integer-like."""

    non_null = series.dropna()

    if non_null.empty:
        return False

    return bool(((non_null % 1) == 0).all())


def _infer_semantic_type(
    series: pd.Series,
    name: str,
    unique_count: int,
    unique_ratio: float,
) -> tuple[SemanticType, float, str]:
    """Infer a semantic type for a single column."""

    non_null = series.dropna()

    if non_null.empty:
        return (
            SemanticType.UNKNOWN,
            1.0,
            "Column contains only missing values.",
        )

    if is_datetime64_any_dtype(series.dtype):
        return (
            SemanticType.DATETIME,
            1.0,
            "Physical dtype is datetime.",
        )

    if is_bool_dtype(series.dtype):
        return (
            SemanticType.BINARY,
            1.0,
            "Physical dtype is boolean.",
        )

    if _looks_like_identifier(name, unique_ratio):
        return (
            SemanticType.IDENTIFIER,
            0.95,
            "Column name suggests an identifier and values are highly unique.",
        )

    if unique_count == 2:
        return (
            SemanticType.BINARY,
            0.95,
            "Column contains exactly two distinct non-null values.",
        )

    if is_numeric_dtype(series.dtype):
        if is_integer_dtype(series.dtype):
            return (
                SemanticType.NUMERIC_DISCRETE,
                0.90,
                "Physical dtype is integer.",
            )

        if is_float_dtype(series.dtype) and _is_integer_like_float(series):
            return (
                SemanticType.NUMERIC_DISCRETE,
                0.85,
                "Floating-point values are integer-like.",
            )

        return (
            SemanticType.NUMERIC_CONTINUOUS,
            0.90,
            "Column contains numeric continuous values.",
        )

    is_textual = (
        is_string_dtype(series.dtype)
        or is_object_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )

    if is_textual:
        if _looks_like_datetime_values(series, name):
            parse_ratio = _datetime_parse_ratio(series)

            if parse_ratio >= 0.80:
                return (
                    SemanticType.DATETIME,
                    parse_ratio,
                    "String values are consistently parseable as datetimes.",
                )

        if isinstance(series.dtype, pd.CategoricalDtype):
            return (
                SemanticType.CATEGORICAL,
                0.95,
                "Physical dtype is categorical.",
            )

        low_cardinality = (unique_count <= 20 and unique_ratio < 0.90) or (
            unique_count <= 50 and unique_ratio <= 0.20
        )

        if low_cardinality:
            return (
                SemanticType.CATEGORICAL,
                0.85,
                "String column has relatively low cardinality.",
            )

        return (
            SemanticType.TEXT,
            0.75,
            "String column has relatively high cardinality.",
        )

    return (
        SemanticType.UNKNOWN,
        0.50,
        "No supported semantic-type rule matched the column.",
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

        semantic_type, confidence, reason = _infer_semantic_type(
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
            )
        )

    return DatasetSchema(
        row_count=row_count,
        column_count=len(data.columns),
        columns=tuple(column_schemas),
    )
