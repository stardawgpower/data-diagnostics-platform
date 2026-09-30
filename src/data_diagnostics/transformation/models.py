from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class RemoveEmptyRows:
    """Remove rows where every value is missing."""


@dataclass(frozen=True, slots=True)
class DropColumns:
    """Drop explicitly selected columns."""

    columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReplaceValueWithMissing:
    """Replace an explicitly selected value with missing data."""

    columns: tuple[str, ...]
    value: object


type TransformationOperation = RemoveEmptyRows | DropColumns | ReplaceValueWithMissing


@dataclass(frozen=True, slots=True)
class TransformationPlan:
    """Ordered collection of user-approved transformations."""

    operations: tuple[TransformationOperation, ...] = ()


@dataclass(frozen=True, slots=True)
class TransformationLogEntry:
    """Audit record for one executed transformation."""

    operation: str
    affected_rows: int
    affected_cells: int
    affected_columns: tuple[str, ...]
    message: str


@dataclass(frozen=True, slots=True)
class TransformationResult:
    """Result produced by applying a transformation plan."""

    data: pd.DataFrame
    original_shape: tuple[int, int]
    final_shape: tuple[int, int]
    log: tuple[TransformationLogEntry, ...]

    @property
    def operation_count(self) -> int:
        """Return the number of executed transformation operations."""

        return len(self.log)
