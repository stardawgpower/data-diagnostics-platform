from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype


class CorrelationMethod(StrEnum):
    """Supported numeric correlation methods."""

    PEARSON = "pearson"
    SPEARMAN = "spearman"


class RelationshipStatus(StrEnum):
    """Availability status for a numeric relationship estimate."""

    OK = "ok"
    INSUFFICIENT_PAIRS = "insufficient_pairs"
    CONSTANT_INPUT = "constant_input"


@dataclass(frozen=True, slots=True)
class NumericRelationship:
    """Correlation summary for two numeric columns."""

    x_column: str
    y_column: str
    method: CorrelationMethod
    total_count: int
    valid_count: int
    excluded_count: int
    minimum_pairs: int
    coefficient: float | None
    status: RelationshipStatus


@dataclass(frozen=True, slots=True)
class CorrelationMatrix:
    """Pairwise correlation coefficients and valid-observation counts."""

    columns: tuple[str, ...]
    method: CorrelationMethod
    minimum_pairs: int
    coefficients: pd.DataFrame
    valid_counts: pd.DataFrame


@dataclass(frozen=True, slots=True)
class CorrelationPair:
    """One valid pairwise correlation extracted from a matrix."""

    x_column: str
    y_column: str
    coefficient: float
    absolute_coefficient: float
    valid_count: int


def _coerce_method(
    method: CorrelationMethod | str,
) -> CorrelationMethod:
    """Normalize a correlation-method argument."""

    if isinstance(method, CorrelationMethod):
        return method

    if not isinstance(method, str):
        raise TypeError("Correlation method must be a string or CorrelationMethod.")

    try:
        return CorrelationMethod(method.lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported correlation method: {method!r}.") from exc


def _validate_minimum_pairs(
    minimum_pairs: int,
) -> None:
    """Validate the minimum number of paired observations."""

    if isinstance(minimum_pairs, bool) or not isinstance(
        minimum_pairs,
        int,
    ):
        raise TypeError("minimum_pairs must be an integer.")

    if minimum_pairs < 2:
        raise ValueError("minimum_pairs must be at least 2.")


def _require_numeric_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a numeric, non-boolean dataset column."""

    if column not in data.columns:
        raise ValueError(f"Unknown column: {column}.")

    series = data[column]

    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        raise ValueError(f"Column {column!r} must be numeric.")

    return series


def _numeric_array(
    data: pd.DataFrame,
    column: str,
) -> np.ndarray:
    """Return a numeric column as a floating-point array."""

    series = _require_numeric_column(
        data,
        column,
    )

    return series.to_numpy(
        dtype=float,
        na_value=np.nan,
    )


def _paired_finite_values(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite observations present in both numeric columns."""

    x_values = _numeric_array(
        data,
        x_column,
    )

    y_values = _numeric_array(
        data,
        y_column,
    )

    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)

    return (
        x_values[valid_mask],
        y_values[valid_mask],
    )


def _is_constant(
    values: np.ndarray,
) -> bool:
    """Return whether all values in a non-empty array are identical."""

    return bool(values.size > 0 and np.unique(values).size < 2)


def _calculate_coefficient(
    x_values: np.ndarray,
    y_values: np.ndarray,
    method: CorrelationMethod,
) -> float:
    """Calculate Pearson or Spearman correlation for finite paired data."""

    if method == CorrelationMethod.SPEARMAN:
        x_values = pd.Series(x_values).rank(method="average").to_numpy(dtype=float)

        y_values = pd.Series(y_values).rank(method="average").to_numpy(dtype=float)

    coefficient = float(
        np.corrcoef(
            x_values,
            y_values,
        )[0, 1]
    )

    return float(
        np.clip(
            coefficient,
            -1.0,
            1.0,
        )
    )


def _relationship_from_values(
    *,
    x_column: str,
    y_column: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    total_count: int,
    method: CorrelationMethod,
    minimum_pairs: int,
) -> NumericRelationship:
    """Build a relationship summary from already paired finite values."""

    valid_count = len(x_values)

    if valid_count < minimum_pairs:
        return NumericRelationship(
            x_column=x_column,
            y_column=y_column,
            method=method,
            total_count=total_count,
            valid_count=valid_count,
            excluded_count=(total_count - valid_count),
            minimum_pairs=minimum_pairs,
            coefficient=None,
            status=RelationshipStatus.INSUFFICIENT_PAIRS,
        )

    if _is_constant(x_values) or _is_constant(y_values):
        return NumericRelationship(
            x_column=x_column,
            y_column=y_column,
            method=method,
            total_count=total_count,
            valid_count=valid_count,
            excluded_count=(total_count - valid_count),
            minimum_pairs=minimum_pairs,
            coefficient=None,
            status=RelationshipStatus.CONSTANT_INPUT,
        )

    coefficient = _calculate_coefficient(
        x_values,
        y_values,
        method,
    )

    return NumericRelationship(
        x_column=x_column,
        y_column=y_column,
        method=method,
        total_count=total_count,
        valid_count=valid_count,
        excluded_count=(total_count - valid_count),
        minimum_pairs=minimum_pairs,
        coefficient=coefficient,
        status=RelationshipStatus.OK,
    )


def analyze_numeric_relationship(
    data: pd.DataFrame,
    x_column: str,
    y_column: str,
    *,
    method: CorrelationMethod | str = CorrelationMethod.PEARSON,
    minimum_pairs: int = 3,
) -> NumericRelationship:
    """
    Estimate correlation between two numeric columns.

    Only rows containing finite observations for both columns are used.
    Correlation is descriptive association and does not imply causation.
    """

    if x_column == y_column:
        raise ValueError("Relationship columns must be different.")

    normalized_method = _coerce_method(
        method,
    )

    _validate_minimum_pairs(
        minimum_pairs,
    )

    x_values, y_values = _paired_finite_values(
        data,
        x_column,
        y_column,
    )

    return _relationship_from_values(
        x_column=x_column,
        y_column=y_column,
        x_values=x_values,
        y_values=y_values,
        total_count=len(data),
        method=normalized_method,
        minimum_pairs=minimum_pairs,
    )


def _validate_matrix_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[str, ...]:
    """Validate columns selected for a correlation matrix."""

    if isinstance(columns, str):
        raise TypeError("columns must be a sequence of column names.")

    column_names = tuple(columns)

    if len(column_names) < 2:
        raise ValueError("At least two numeric columns are required.")

    if len(set(column_names)) != len(column_names):
        raise ValueError("Correlation matrix columns must be unique.")

    for column in column_names:
        _require_numeric_column(
            data,
            column,
        )

    return column_names


def build_correlation_matrix(
    data: pd.DataFrame,
    columns: Sequence[str],
    *,
    method: CorrelationMethod | str = CorrelationMethod.PEARSON,
    minimum_pairs: int = 3,
) -> CorrelationMatrix:
    """
    Build a pairwise numeric correlation matrix.

    Each pair uses only rows where both values are finite. A separate
    valid-count matrix records the sample size used for each coefficient.
    """

    normalized_method = _coerce_method(
        method,
    )

    _validate_minimum_pairs(
        minimum_pairs,
    )

    column_names = _validate_matrix_columns(
        data,
        columns,
    )

    size = len(column_names)

    coefficients = pd.DataFrame(
        np.nan,
        index=column_names,
        columns=column_names,
        dtype=float,
    )

    valid_counts = pd.DataFrame(
        0,
        index=column_names,
        columns=column_names,
        dtype=int,
    )

    arrays = {
        column: _numeric_array(
            data,
            column,
        )
        for column in column_names
    }

    for index, x_column in enumerate(column_names):
        x_array = arrays[x_column]

        diagonal_values = x_array[np.isfinite(x_array)]

        diagonal_count = len(diagonal_values)

        valid_counts.loc[
            x_column,
            x_column,
        ] = diagonal_count

        if diagonal_count >= minimum_pairs and not _is_constant(diagonal_values):
            coefficients.loc[
                x_column,
                x_column,
            ] = 1.0

        for offset in range(
            index + 1,
            size,
        ):
            y_column = column_names[offset]

            y_array = arrays[y_column]

            valid_mask = np.isfinite(x_array) & np.isfinite(y_array)

            x_values = x_array[valid_mask]

            y_values = y_array[valid_mask]

            relationship = _relationship_from_values(
                x_column=x_column,
                y_column=y_column,
                x_values=x_values,
                y_values=y_values,
                total_count=len(data),
                method=normalized_method,
                minimum_pairs=minimum_pairs,
            )

            valid_counts.loc[
                x_column,
                y_column,
            ] = relationship.valid_count

            valid_counts.loc[
                y_column,
                x_column,
            ] = relationship.valid_count

            if relationship.coefficient is not None:
                coefficients.loc[
                    x_column,
                    y_column,
                ] = relationship.coefficient

                coefficients.loc[
                    y_column,
                    x_column,
                ] = relationship.coefficient

    return CorrelationMatrix(
        columns=column_names,
        method=normalized_method,
        minimum_pairs=minimum_pairs,
        coefficients=coefficients,
        valid_counts=valid_counts,
    )


def strongest_correlations(
    matrix: CorrelationMatrix,
    *,
    top_n: int = 10,
    minimum_absolute: float = 0.0,
) -> tuple[CorrelationPair, ...]:
    """
    Return valid pairwise correlations ordered by absolute magnitude.

    Only the upper triangle of the matrix is considered, so each pair
    appears once. Ordering describes correlation magnitude only.
    """

    if isinstance(top_n, bool) or not isinstance(
        top_n,
        int,
    ):
        raise TypeError("top_n must be an integer.")

    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    if isinstance(minimum_absolute, bool):
        raise TypeError("minimum_absolute must be numeric.")

    try:
        threshold = float(minimum_absolute)
    except (TypeError, ValueError) as exc:
        raise TypeError("minimum_absolute must be numeric.") from exc

    if not np.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError("minimum_absolute must be between 0 and 1.")

    relationships = []

    for index, x_column in enumerate(matrix.columns):
        for y_column in matrix.columns[index + 1 :]:
            coefficient = matrix.coefficients.loc[
                x_column,
                y_column,
            ]

            if pd.isna(coefficient):
                continue

            coefficient_value = float(coefficient)

            absolute_coefficient = abs(coefficient_value)

            if absolute_coefficient < threshold:
                continue

            relationships.append(
                CorrelationPair(
                    x_column=x_column,
                    y_column=y_column,
                    coefficient=coefficient_value,
                    absolute_coefficient=absolute_coefficient,
                    valid_count=int(
                        matrix.valid_counts.loc[
                            x_column,
                            y_column,
                        ]
                    ),
                )
            )

    relationships.sort(
        key=lambda relationship: (
            -relationship.absolute_coefficient,
            relationship.x_column,
            relationship.y_column,
        )
    )

    return tuple(relationships[:top_n])
