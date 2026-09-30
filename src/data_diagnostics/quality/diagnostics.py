import math
import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
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
    ALL_MISSING_ROWS = "all_missing_rows"
    CONSTANT = "constant"
    HIGH_MISSINGNESS = "high_missingness"
    HIGH_CARDINALITY = "high_cardinality"
    PROBABLE_IDENTIFIER = "probable_identifier"
    DUPLICATE_ROWS = "duplicate_rows"
    STRUCTURAL_EMPTY_COLUMN = "structural_empty_column"
    POSSIBLE_SENTINEL = "possible_sentinel"
    NON_FINITE_VALUES = "non_finite_values"


@dataclass(frozen=True, slots=True)
class QualityIssue:
    """One detected data-quality issue."""

    code: QualityIssueCode
    severity: QualitySeverity
    message: str
    column: str | None = None
    observed_value: object | None = None
    affected_count: int | None = None
    affected_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class DataQualityReport:
    """Summary of data-quality characteristics for a dataset."""

    row_count: int
    column_count: int
    missing_cell_count: int
    missing_cell_ratio: float
    all_missing_row_count: int
    all_missing_row_ratio: float
    duplicate_row_count: int
    duplicate_row_ratio: float
    issues: tuple[QualityIssue, ...]

    @property
    def issue_count(self) -> int:
        """Return the number of detected issues."""

        return len(self.issues)


_UNNAMED_COLUMN_PATTERN = re.compile(
    r"^Unnamed:\s*\d+$",
    re.IGNORECASE,
)


def _detect_possible_sentinel(
    series: pd.Series,
) -> tuple[float, int, float] | None:
    """
    Detect a repeated extreme numeric value that may encode missing data.

    The value is only flagged. It is never replaced automatically.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    numeric = numeric[np.isfinite(numeric)]

    if len(numeric) < 10 or numeric.nunique() < 3:
        return None

    minimum_count = max(
        5,
        math.ceil(len(numeric) * 0.01),
    )

    candidates = {
        float(numeric.min()),
        float(numeric.max()),
    }

    best_candidate: tuple[float, int, float, float] | None = None

    for candidate in candidates:
        candidate_count = int((numeric == candidate).sum())

        if candidate_count < minimum_count:
            continue

        remaining = numeric[numeric != candidate]

        if len(remaining) < 3:
            continue

        median = float(remaining.median())
        mad = float((remaining - median).abs().median())

        if mad > 0:
            distance_score = abs(candidate - median) / (1.4826 * mad)
        else:
            standard_deviation = float(remaining.std())

            if standard_deviation <= 0 or pd.isna(standard_deviation):
                continue

            distance_score = abs(candidate - median) / standard_deviation

        if distance_score < 2.5:
            continue

        ratio = candidate_count / len(numeric)
        detected = (
            candidate,
            candidate_count,
            ratio,
            distance_score,
        )

        if best_candidate is None or detected[3] > best_candidate[3]:
            best_candidate = detected

    if best_candidate is None:
        return None

    return (
        best_candidate[0],
        best_candidate[1],
        best_candidate[2],
    )


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

    all_missing_row_mask = data.isna().all(axis=1)
    all_missing_row_count = int(all_missing_row_mask.sum())
    all_missing_row_ratio = all_missing_row_count / row_count if row_count else 0.0

    populated_data = data.loc[~all_missing_row_mask]
    duplicate_row_count = int(populated_data.duplicated().sum())
    duplicate_row_ratio = duplicate_row_count / len(populated_data) if len(populated_data) else 0.0

    schema = infer_schema(data)
    issues: list[QualityIssue] = []

    if all_missing_row_count > 0:
        issues.append(
            QualityIssue(
                code=QualityIssueCode.ALL_MISSING_ROWS,
                severity=QualitySeverity.WARNING,
                message=(
                    f"{all_missing_row_count} completely empty row(s) detected "
                    f"({all_missing_row_ratio:.1%} of rows)."
                ),
                affected_count=all_missing_row_count,
                affected_ratio=all_missing_row_ratio,
            )
        )

    if duplicate_row_count > 0:
        issues.append(
            QualityIssue(
                code=QualityIssueCode.DUPLICATE_ROWS,
                severity=QualitySeverity.WARNING,
                message=(
                    f"{duplicate_row_count} duplicate populated row(s) detected "
                    f"({duplicate_row_ratio:.1%} of populated rows)."
                ),
                affected_count=duplicate_row_count,
                affected_ratio=duplicate_row_ratio,
            )
        )

    for column in schema.columns:
        all_missing = row_count > 0 and column.missing_count == row_count

        if all_missing:
            if _UNNAMED_COLUMN_PATTERN.fullmatch(column.name):
                issues.append(
                    QualityIssue(
                        code=QualityIssueCode.STRUCTURAL_EMPTY_COLUMN,
                        severity=QualitySeverity.INFO,
                        column=column.name,
                        message=(
                            "Unnamed column contains only missing values and may "
                            "have been created by a trailing delimiter."
                        ),
                    )
                )
            else:
                issues.append(
                    QualityIssue(
                        code=QualityIssueCode.ALL_MISSING,
                        severity=QualitySeverity.WARNING,
                        column=column.name,
                        message="Column contains only missing values.",
                    )
                )

            continue

        series = data[column.name]

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            non_finite_mask = numeric.notna() & ~np.isfinite(numeric)
            non_finite_count = int(non_finite_mask.sum())

            if non_finite_count > 0:
                issues.append(
                    QualityIssue(
                        code=QualityIssueCode.NON_FINITE_VALUES,
                        severity=QualitySeverity.WARNING,
                        column=column.name,
                        message=(
                            f"Column contains {non_finite_count} infinite or "
                            "non-finite numeric value(s)."
                        ),
                        affected_count=non_finite_count,
                    )
                )

            sentinel = _detect_possible_sentinel(series)

            if sentinel is not None:
                sentinel_value, sentinel_count, sentinel_ratio = sentinel

                issues.append(
                    QualityIssue(
                        code=QualityIssueCode.POSSIBLE_SENTINEL,
                        severity=QualitySeverity.INFO,
                        column=column.name,
                        message=(
                            f"Extreme value {sentinel_value:g} occurs "
                            f"{sentinel_count} time(s) ({sentinel_ratio:.1%}). "
                            "It may represent a sentinel or encoded missing "
                            "value and should be reviewed."
                        ),
                        observed_value=sentinel_value,
                        affected_count=sentinel_count,
                        affected_ratio=sentinel_ratio,
                    )
                )

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
                    message=f"Column has {column.missing_ratio:.1%} missing values.",
                    affected_count=column.missing_count,
                    affected_ratio=column.missing_ratio,
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
        all_missing_row_count=all_missing_row_count,
        all_missing_row_ratio=all_missing_row_ratio,
        duplicate_row_count=duplicate_row_count,
        duplicate_row_ratio=duplicate_row_ratio,
        issues=tuple(issues),
    )
