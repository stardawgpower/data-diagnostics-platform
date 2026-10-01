from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)


class PivotAggregation(StrEnum):
    """Supported pivot aggregation operations."""

    ROW_COUNT = "row_count"
    COUNT = "count"
    SUM = "sum"
    MEAN = "mean"
    MEDIAN = "median"
    MINIMUM = "min"
    MAXIMUM = "max"


@dataclass(frozen=True, slots=True)
class PivotCardinality:
    """Potential dense output size for a pivot configuration."""

    row_group_count: int
    column_group_count: int
    potential_cell_count: int
    max_cells: int


@dataclass(frozen=True, slots=True)
class PivotResult:
    """Grouped and matrix representations of a pivot analysis."""

    row_dimensions: tuple[str, ...]
    column_dimensions: tuple[str, ...]
    value_column: str | None
    aggregation: PivotAggregation
    include_missing_dimensions: bool
    source_row_count: int
    included_row_count: int
    excluded_dimension_row_count: int
    group_count: int
    cardinality: PivotCardinality
    tidy: pd.DataFrame
    table: pd.DataFrame


def _normalize_columns(
    columns: Sequence[str],
    *,
    name: str,
) -> tuple[str, ...]:
    """Normalize and validate a dimension-column collection."""

    normalized = tuple(columns)

    if any(not isinstance(column, str) for column in normalized):
        raise TypeError(f"{name} must contain column names as strings.")

    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicate columns.")

    return normalized


def _require_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a dataset column or raise a clear validation error."""

    if column not in data.columns:
        raise ValueError(f"Unknown column: {column}.")

    return data[column]


def _require_columns(
    data: pd.DataFrame,
    columns: Sequence[str],
) -> None:
    """Validate that all requested columns exist."""

    for column in columns:
        _require_column(
            data,
            column,
        )


def _coerce_aggregation(
    aggregation: PivotAggregation | str,
) -> PivotAggregation:
    """Normalize a pivot aggregation."""

    if isinstance(
        aggregation,
        PivotAggregation,
    ):
        return aggregation

    if not isinstance(
        aggregation,
        str,
    ):
        raise TypeError("aggregation must be a string or PivotAggregation.")

    try:
        return PivotAggregation(aggregation.lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported pivot aggregation: {aggregation!r}.") from exc


def _validate_max_cells(
    max_cells: int,
) -> int:
    """Validate the dense-output safety limit."""

    if isinstance(
        max_cells,
        bool,
    ) or not isinstance(
        max_cells,
        int,
    ):
        raise TypeError("max_cells must be an integer.")

    if max_cells < 1:
        raise ValueError("max_cells must be at least 1.")

    return max_cells


def _validate_configuration(
    data: pd.DataFrame,
    *,
    row_dimensions: tuple[str, ...],
    column_dimensions: tuple[str, ...],
    value_column: str | None,
    aggregation: PivotAggregation,
) -> None:
    """Validate dimension and measure compatibility."""

    if not row_dimensions:
        raise ValueError("At least one row dimension is required.")

    overlap = set(row_dimensions) & set(column_dimensions)

    if overlap:
        overlap_text = ", ".join(sorted(overlap))

        raise ValueError(
            f"Row and column dimensions must be distinct. Overlapping columns: {overlap_text}."
        )

    dimension_columns = row_dimensions + column_dimensions

    _require_columns(
        data,
        dimension_columns,
    )

    if aggregation == PivotAggregation.ROW_COUNT:
        if value_column is not None:
            raise ValueError("value_column must be None when aggregation is row_count.")

        return

    if value_column is None:
        raise ValueError("value_column is required for this aggregation.")

    _require_column(
        data,
        value_column,
    )

    if value_column in dimension_columns:
        raise ValueError("value_column must be different from all pivot dimensions.")

    if aggregation == PivotAggregation.COUNT:
        return

    value_series = data[value_column]

    if is_bool_dtype(value_series.dtype) or not is_numeric_dtype(value_series.dtype):
        raise ValueError(
            f"Column {value_column!r} must be numeric for {aggregation.value!r} aggregation."
        )


def _included_data(
    data: pd.DataFrame,
    dimension_columns: tuple[str, ...],
    *,
    include_missing_dimensions: bool,
) -> tuple[pd.DataFrame, int]:
    """Return rows eligible for grouping and their exclusion count."""

    if include_missing_dimensions:
        return (
            data.copy(),
            0,
        )

    valid_mask = data[list(dimension_columns)].notna().all(axis=1)

    included = data.loc[valid_mask].copy()

    excluded_count = len(data) - len(included)

    return (
        included,
        excluded_count,
    )


def _group_count(
    data: pd.DataFrame,
    columns: tuple[str, ...],
    *,
    include_missing_dimensions: bool,
) -> int:
    """Count observed unique dimension combinations."""

    if not columns:
        return 1

    if data.empty:
        return 0

    grouped = data.groupby(
        list(columns),
        dropna=not include_missing_dimensions,
        observed=True,
        sort=True,
    )

    return int(grouped.ngroups)


def estimate_pivot_cardinality(
    data: pd.DataFrame,
    row_dimensions: Sequence[str],
    column_dimensions: Sequence[str] = (),
    *,
    include_missing_dimensions: bool = False,
    max_cells: int = 10_000,
) -> PivotCardinality:
    """
    Estimate the potential dense matrix size for a pivot configuration.

    Cardinality is based on observed row-dimension and column-dimension
    combinations after applying the requested missing-dimension policy.
    """

    normalized_rows = _normalize_columns(
        row_dimensions,
        name="row_dimensions",
    )

    normalized_columns = _normalize_columns(
        column_dimensions,
        name="column_dimensions",
    )

    if not normalized_rows:
        raise ValueError("At least one row dimension is required.")

    overlap = set(normalized_rows) & set(normalized_columns)

    if overlap:
        raise ValueError("Row and column dimensions must be distinct.")

    dimension_columns = normalized_rows + normalized_columns

    _require_columns(
        data,
        dimension_columns,
    )

    validated_max_cells = _validate_max_cells(max_cells)

    included, _ = _included_data(
        data,
        dimension_columns,
        include_missing_dimensions=include_missing_dimensions,
    )

    row_group_count = _group_count(
        included,
        normalized_rows,
        include_missing_dimensions=include_missing_dimensions,
    )

    if normalized_columns:
        column_group_count = _group_count(
            included,
            normalized_columns,
            include_missing_dimensions=include_missing_dimensions,
        )

    else:
        column_group_count = 1

    potential_cell_count = row_group_count * column_group_count

    return PivotCardinality(
        row_group_count=row_group_count,
        column_group_count=column_group_count,
        potential_cell_count=potential_cell_count,
        max_cells=validated_max_cells,
    )


def _aggregate_groups(
    data: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    value_column: str | None,
    aggregation: PivotAggregation,
    include_missing_dimensions: bool,
) -> pd.Series:
    """Aggregate observations by the requested pivot dimensions."""

    grouped = data.groupby(
        list(group_columns),
        dropna=not include_missing_dimensions,
        observed=True,
        sort=True,
    )

    if aggregation == PivotAggregation.ROW_COUNT:
        result = grouped.size()

    elif aggregation == PivotAggregation.COUNT:
        result = grouped[value_column].count()

    elif aggregation == PivotAggregation.SUM:
        result = grouped[value_column].sum(min_count=1)

    elif aggregation == PivotAggregation.MEAN:
        result = grouped[value_column].mean()

    elif aggregation == PivotAggregation.MEDIAN:
        result = grouped[value_column].median()

    elif aggregation == PivotAggregation.MINIMUM:
        result = grouped[value_column].min()

    else:
        result = grouped[value_column].max()

    result.name = "value"

    return result


def build_pivot_table(
    data: pd.DataFrame,
    row_dimensions: Sequence[str],
    column_dimensions: Sequence[str] = (),
    *,
    value_column: str | None = None,
    aggregation: PivotAggregation | str = PivotAggregation.ROW_COUNT,
    include_missing_dimensions: bool = False,
    max_cells: int = 10_000,
) -> PivotResult:
    """
    Build a generic pivot analysis without modifying the input dataset.

    The result contains both a tidy grouped representation and a pivoted
    matrix. Missing dimension values are excluded by default. Missing
    combinations in the matrix remain missing rather than being filled
    automatically.
    """

    normalized_rows = _normalize_columns(
        row_dimensions,
        name="row_dimensions",
    )

    normalized_columns = _normalize_columns(
        column_dimensions,
        name="column_dimensions",
    )

    normalized_aggregation = _coerce_aggregation(aggregation)

    _validate_configuration(
        data,
        row_dimensions=normalized_rows,
        column_dimensions=normalized_columns,
        value_column=value_column,
        aggregation=normalized_aggregation,
    )

    dimension_columns = normalized_rows + normalized_columns

    included, excluded_count = _included_data(
        data,
        dimension_columns,
        include_missing_dimensions=include_missing_dimensions,
    )

    cardinality = estimate_pivot_cardinality(
        data,
        normalized_rows,
        normalized_columns,
        include_missing_dimensions=include_missing_dimensions,
        max_cells=max_cells,
    )

    if cardinality.potential_cell_count > cardinality.max_cells:
        raise ValueError(
            "Pivot configuration is too large: "
            f"{cardinality.potential_cell_count:,} potential cells "
            f"exceeds the limit of {cardinality.max_cells:,}."
        )

    group_columns = normalized_rows + normalized_columns

    if included.empty:
        tidy_columns = [
            *group_columns,
            "value",
        ]

        tidy = pd.DataFrame(columns=tidy_columns)

        table = pd.DataFrame()

    else:
        grouped_values = _aggregate_groups(
            included,
            group_columns=group_columns,
            value_column=value_column,
            aggregation=normalized_aggregation,
            include_missing_dimensions=include_missing_dimensions,
        )

        tidy = grouped_values.reset_index().copy()

        if normalized_columns:
            table = grouped_values.unstack(list(normalized_columns))

            if isinstance(
                table,
                pd.Series,
            ):
                table = table.to_frame(name="value")

            else:
                table = table.copy()

        else:
            table = grouped_values.to_frame(name="value")

    return PivotResult(
        row_dimensions=normalized_rows,
        column_dimensions=normalized_columns,
        value_column=value_column,
        aggregation=normalized_aggregation,
        include_missing_dimensions=include_missing_dimensions,
        source_row_count=len(data),
        included_row_count=len(included),
        excluded_dimension_row_count=excluded_count,
        group_count=len(tidy),
        cardinality=cardinality,
        tidy=tidy,
        table=table,
    )
