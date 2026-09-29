from dataclasses import dataclass
from enum import Enum

import pandas as pd

from data_diagnostics.ingestion.schema import SemanticType, infer_schema


class QualitySeverity(str, Enum):
    """Severity assigned to a data-quality issue."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class QualityIssueCode(str, Enum):
    """Machine-readable identifiers for data-quality issues."""

    ALL_MISSING = "all_missing"
    CONSTANT = "constant"
    HIGH_MISSINGNESS = "high_missingness"
    HIGH_CARDINALITY = "high_cardinality"
    PROBABLE_IDENTIFIER = "probable_identifier"
    DUPLICATE_ROWS = "duplicate_rows"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One detected data-quality issue."""

    code: QualityIssueCode
    severity: QualitySeverity
    message: str
    column: str | None = None


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Summary of data-quality characteristics for a dataset."""

    row_count: int
    column_count: int
    missing_cell_count: int
    missing_cell_ratio: float
    duplicate_row_count: int
    duplicate_row_ratio: float
    issues: tuple[QualityIssue, ...]

    @property
    def issue_count(self) -> int:
        """Return the number of detected issues."""

        return len(self.issues)


def analyze_data_quality(
    data: pd.DataFrame,
    *,
    high_missingness_threshold: float = 0.40,
    high_cardinality_ratio: float = 0.80,
    high_cardinality_min_unique: int = 20,
) -> DataQualityReport:
    """
    Analyze a DataFrame for common data-quality issues.

    The input DataFrame is never modified.
    """

    if not 0 <= high_missingness_threshold <= 1:
        raise ValueError("high_missingness_threshold must be between 0 and 1.")

    if not 0 <= high_cardinality_ratio <= 1:
        raise ValueError("high_cardinality_ratio must be between 0 and 1.")

    if high_cardinality_min_unique < 1:
        raise ValueError("high_cardinality_min_unique must be at least 1.")

    row_count = len(data)
    column_count = len(data.columns)
    total_cells = row_count * column_count

    missing_cell_count = int(data.isna().sum().sum())
    missing_cell_ratio = missing_cell_count / total_cells if total_cells else 0.0

    duplicate_row_count = int(data.duplicated().sum())
    duplicate_row_ratio = duplicate_row_count / row_count if row_count else 0.0

    schema = infer_schema(data)
    issues: list[QualityIssue] = []

    if duplicate_row_count > 0:
        issues.append(
            QualityIssue(
                code=QualityIssueCode.DUPLICATE_ROWS,
                severity=QualitySeverity.WARNING,
                message=(
                    f"{duplicate_row_count} duplicate row(s) detected "
                    f"({duplicate_row_ratio:.1%} of rows)."
                ),
            )
        )

    for column in schema.columns:
        all_missing = row_count > 0 and column.missing_count == row_count

        if all_missing:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.ALL_MISSING,
                    severity=QualitySeverity.WARNING,
                    column=column.name,
                    message="Column contains only missing values.",
                )
            )
            continue

        if column.unique_count == 1:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.CONSTANT,
                    severity=QualitySeverity.WARNING,
                    column=column.name,
                    message="Column contains only one distinct non-null value.",
                )
            )

        if column.missing_ratio >= high_missingness_threshold:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.HIGH_MISSINGNESS,
                    severity=QualitySeverity.WARNING,
                    column=column.name,
                    message=(f"Column has {column.missing_ratio:.1%} missing values."),
                )
            )

        if column.semantic_type == SemanticType.IDENTIFIER:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.PROBABLE_IDENTIFIER,
                    severity=QualitySeverity.INFO,
                    column=column.name,
                    message=(
                        "Column appears to be an identifier and may not be "
                        "appropriate as a predictive feature."
                    ),
                )
            )

        high_cardinality = (
            column.semantic_type
            in {
                SemanticType.CATEGORICAL,
                SemanticType.TEXT,
            }
            and column.unique_count >= high_cardinality_min_unique
            and column.unique_ratio >= high_cardinality_ratio
        )

        if high_cardinality:
            issues.append(
                QualityIssue(
                    code=QualityIssueCode.HIGH_CARDINALITY,
                    severity=QualitySeverity.INFO,
                    column=column.name,
                    message=(
                        f"Column has {column.unique_count} distinct values "
                        f"({column.unique_ratio:.1%} of non-null rows)."
                    ),
                )
            )

    return DataQualityReport(
        row_count=row_count,
        column_count=column_count,
        missing_cell_count=missing_cell_count,
        missing_cell_ratio=missing_cell_ratio,
        duplicate_row_count=duplicate_row_count,
        duplicate_row_ratio=duplicate_row_ratio,
        issues=tuple(issues),
    )
