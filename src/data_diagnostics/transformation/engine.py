import pandas as pd
from pandas.api.types import is_scalar

from data_diagnostics.transformation.exceptions import (
    InvalidTransformationError,
    UnknownColumnError,
)
from data_diagnostics.transformation.models import (
    DropColumns,
    RemoveEmptyRows,
    ReplaceValueWithMissing,
    TransformationLogEntry,
    TransformationPlan,
    TransformationResult,
)


def _validate_columns(
    data: pd.DataFrame,
    columns: tuple[str, ...],
) -> None:
    """Ensure requested columns exist and are not duplicated."""

    if not columns:
        raise InvalidTransformationError("At least one column must be selected.")

    if len(set(columns)) != len(columns):
        raise InvalidTransformationError("A transformation cannot contain duplicate column names.")

    missing_columns = [column for column in columns if column not in data.columns]

    if missing_columns:
        formatted = ", ".join(missing_columns)

        raise UnknownColumnError(f"Unknown column(s): {formatted}.")


def _remove_empty_rows(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, TransformationLogEntry]:
    """Remove rows where every cell is missing."""

    empty_row_mask = data.isna().all(axis=1)

    affected_rows = int(empty_row_mask.sum())
    affected_cells = affected_rows * len(data.columns)

    transformed = data.loc[~empty_row_mask].copy()

    log_entry = TransformationLogEntry(
        operation="remove_empty_rows",
        affected_rows=affected_rows,
        affected_cells=affected_cells,
        affected_columns=tuple(str(column) for column in data.columns),
        message=(f"Removed {affected_rows} completely empty row(s)."),
    )

    return transformed, log_entry


def _drop_columns(
    data: pd.DataFrame,
    operation: DropColumns,
) -> tuple[pd.DataFrame, TransformationLogEntry]:
    """Drop explicitly selected columns."""

    _validate_columns(
        data,
        operation.columns,
    )

    remaining_column_count = len(data.columns) - len(operation.columns)

    if remaining_column_count == 0:
        raise InvalidTransformationError(
            "The transformation would remove every column from the working dataset."
        )

    affected_rows = len(data)
    affected_cells = affected_rows * len(operation.columns)

    transformed = data.drop(columns=list(operation.columns)).copy()

    log_entry = TransformationLogEntry(
        operation="drop_columns",
        affected_rows=affected_rows,
        affected_cells=affected_cells,
        affected_columns=operation.columns,
        message=("Dropped column(s): " + ", ".join(operation.columns) + "."),
    )

    return transformed, log_entry


def _replace_value_with_missing(
    data: pd.DataFrame,
    operation: ReplaceValueWithMissing,
) -> tuple[pd.DataFrame, TransformationLogEntry]:
    """Replace an approved scalar value with missing data."""

    _validate_columns(
        data,
        operation.columns,
    )

    if not is_scalar(operation.value):
        raise InvalidTransformationError("Replacement values must be scalar values.")

    if pd.isna(operation.value):
        raise InvalidTransformationError("The selected replacement value is already missing.")

    transformed = data.copy(deep=True)

    selected = transformed.loc[
        :,
        list(operation.columns),
    ]

    match_mask = selected.eq(operation.value).fillna(False)

    affected_cells = int(match_mask.sum().sum())

    affected_rows = int(match_mask.any(axis=1).sum())

    for column in operation.columns:
        transformed[column] = transformed[column].mask(
            match_mask[column],
            pd.NA,
        )

    log_entry = TransformationLogEntry(
        operation="replace_value_with_missing",
        affected_rows=affected_rows,
        affected_cells=affected_cells,
        affected_columns=operation.columns,
        message=(
            f"Replaced value {operation.value!r} with missing "
            f"in {len(operation.columns)} column(s); "
            f"{affected_cells} cell(s) affected."
        ),
    )

    return transformed, log_entry


def apply_transformation_plan(
    data: pd.DataFrame,
    plan: TransformationPlan,
) -> TransformationResult:
    """
    Apply an ordered transformation plan to a copy of a DataFrame.

    The source DataFrame is never modified.
    """

    if not isinstance(data, pd.DataFrame):
        raise InvalidTransformationError("Transformation input must be a pandas DataFrame.")

    if not isinstance(plan, TransformationPlan):
        raise InvalidTransformationError("A valid TransformationPlan is required.")

    original_shape = data.shape

    working_data = data.copy(deep=True)

    log: list[TransformationLogEntry] = []

    for operation in plan.operations:
        if isinstance(
            operation,
            RemoveEmptyRows,
        ):
            working_data, entry = _remove_empty_rows(working_data)

        elif isinstance(
            operation,
            DropColumns,
        ):
            working_data, entry = _drop_columns(
                working_data,
                operation,
            )

        elif isinstance(
            operation,
            ReplaceValueWithMissing,
        ):
            working_data, entry = _replace_value_with_missing(
                working_data,
                operation,
            )

        else:
            raise InvalidTransformationError(
                f"Unsupported transformation operation: {type(operation).__name__}."
            )

        log.append(entry)

    return TransformationResult(
        data=working_data,
        original_shape=original_shape,
        final_shape=working_data.shape,
        log=tuple(log),
    )
