from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype
from scipy.stats import (
    chi2_contingency,
    f_oneway,
    kruskal,
    mannwhitneyu,
    ttest_ind,
)


class TwoGroupMethod(StrEnum):
    """Supported methods for comparing two independent numeric groups."""

    WELCH_T = "welch_t"
    MANN_WHITNEY = "mann_whitney"


class MultiGroupMethod(StrEnum):
    """Supported methods for comparing multiple independent numeric groups."""

    ONE_WAY_ANOVA = "one_way_anova"
    KRUSKAL_WALLIS = "kruskal_wallis"


@dataclass(frozen=True, slots=True)
class NumericGroupSummary:
    """Descriptive summary for one numeric group."""

    group: object
    count: int
    mean: float
    median: float
    standard_deviation: float | None
    minimum: float
    maximum: float


@dataclass(frozen=True, slots=True)
class TwoGroupComparison:
    """Result of an independent two-group numeric comparison."""

    value_column: str
    group_column: str
    group_a: object
    group_b: object
    method: TwoGroupMethod
    alpha: float
    source_row_count: int
    selected_row_count: int
    used_row_count: int
    excluded_value_row_count: int
    group_a_summary: NumericGroupSummary
    group_b_summary: NumericGroupSummary
    statistic: float
    p_value: float
    significant: bool | None
    effect_size_name: str
    effect_size: float | None


@dataclass(frozen=True, slots=True)
class MultiGroupComparison:
    """Result of an independent multi-group numeric comparison."""

    value_column: str
    group_column: str
    groups: tuple[object, ...]
    method: MultiGroupMethod
    alpha: float
    source_row_count: int
    selected_row_count: int
    used_row_count: int
    excluded_value_row_count: int
    group_summaries: tuple[NumericGroupSummary, ...]
    statistic: float
    p_value: float
    significant: bool | None
    effect_size_name: str
    effect_size: float | None


@dataclass(frozen=True, slots=True)
class CategoricalAssociation:
    """Pearson chi-square association result for two categorical variables."""

    row_column: str
    column_column: str
    alpha: float
    source_row_count: int
    used_row_count: int
    excluded_missing_row_count: int
    row_level_count: int
    column_level_count: int
    cell_count: int
    statistic: float
    degrees_of_freedom: int
    p_value: float
    significant: bool | None
    cramers_v: float | None
    minimum_expected_frequency: float
    expected_below_five_count: int
    expected_below_five_ratio: float
    observed: pd.DataFrame
    expected: pd.DataFrame


def _require_column(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Return a column or raise a clear validation error."""

    if column not in data.columns:
        raise ValueError(f"Unknown column: {column}.")

    return data[column]


def _require_distinct_columns(
    first: str,
    second: str,
) -> None:
    """Require two analysis columns to be different."""

    if first == second:
        raise ValueError("Analysis columns must be distinct.")


def _require_numeric_measure(
    data: pd.DataFrame,
    column: str,
) -> pd.Series:
    """Validate and return a numeric, non-boolean measure column."""

    series = _require_column(
        data,
        column,
    )

    if is_bool_dtype(series.dtype) or not is_numeric_dtype(series.dtype):
        raise ValueError(f"Column {column!r} must be numeric and non-boolean.")

    return series


def _validate_alpha(
    alpha: float,
) -> float:
    """Validate a significance threshold."""

    if isinstance(alpha, bool) or not isinstance(
        alpha,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError("alpha must be numeric.")

    normalized = float(alpha)

    if not 0.0 < normalized < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    return normalized


def _validate_max_cells(
    max_cells: int,
) -> int:
    """Validate a contingency-table safety limit."""

    if isinstance(max_cells, bool) or not isinstance(
        max_cells,
        (int, np.integer),
    ):
        raise TypeError("max_cells must be an integer.")

    normalized = int(max_cells)

    if normalized < 1:
        raise ValueError("max_cells must be at least 1.")

    return normalized


def _coerce_two_group_method(
    method: TwoGroupMethod | str,
) -> TwoGroupMethod:
    """Normalize a two-group test method."""

    if isinstance(method, TwoGroupMethod):
        return method

    if not isinstance(method, str):
        raise TypeError("method must be a string or TwoGroupMethod.")

    try:
        return TwoGroupMethod(method.lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported two-group method: {method!r}.") from exc


def _coerce_multi_group_method(
    method: MultiGroupMethod | str,
) -> MultiGroupMethod:
    """Normalize a multi-group test method."""

    if isinstance(method, MultiGroupMethod):
        return method

    if not isinstance(method, str):
        raise TypeError("method must be a string or MultiGroupMethod.")

    try:
        return MultiGroupMethod(method.lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported multi-group method: {method!r}.") from exc


def _validate_group_value(
    value: object,
    *,
    name: str,
) -> None:
    """Validate an explicitly selected group label."""

    if not isinstance(value, Hashable):
        raise TypeError(f"{name} must be hashable.")

    try:
        missing = bool(pd.isna(value))
    except (TypeError, ValueError):
        missing = False

    if missing:
        raise ValueError(f"{name} must not be missing.")


def _validate_group_values(
    group_values: Sequence[object],
) -> tuple[object, ...]:
    """Validate explicitly selected multi-group labels."""

    normalized = tuple(group_values)

    if len(normalized) < 2:
        raise ValueError("At least two group values are required.")

    for index, value in enumerate(normalized):
        _validate_group_value(
            value,
            name=f"group_values[{index}]",
        )

    if len(set(normalized)) != len(normalized):
        raise ValueError("group_values must not contain duplicates.")

    return normalized


def _numeric_values(
    series: pd.Series,
) -> np.ndarray:
    """Convert a validated numeric series to floating-point values."""

    return series.to_numpy(
        dtype=float,
        na_value=np.nan,
    )


def _safe_significance(
    p_value: float,
    alpha: float,
) -> bool | None:
    """Return significance when the p-value is finite."""

    if not np.isfinite(p_value):
        return None

    return bool(p_value < alpha)


def _summarize_group(
    group: object,
    values: np.ndarray,
) -> NumericGroupSummary:
    """Create descriptive statistics for one finite numeric group."""

    count = len(values)

    if count == 0:
        raise ValueError(f"Group {group!r} has no finite numeric observations.")

    standard_deviation = float(np.std(values, ddof=1)) if count > 1 else None

    return NumericGroupSummary(
        group=group,
        count=count,
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        standard_deviation=standard_deviation,
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
    )


def _extract_group_values(
    data: pd.DataFrame,
    *,
    numeric_values: np.ndarray,
    group_column: str,
    group_value: object,
) -> np.ndarray:
    """Return finite numeric values belonging to one explicit group."""

    group_series = data[group_column]

    group_mask = group_series.eq(group_value).fillna(False).to_numpy(dtype=bool)

    finite_mask = np.isfinite(numeric_values)

    return numeric_values[group_mask & finite_mask]


def _hedges_g(
    first: np.ndarray,
    second: np.ndarray,
) -> float | None:
    """Compute Hedges' g using pooled sample variance."""

    first_count = len(first)
    second_count = len(second)

    if first_count < 2 or second_count < 2:
        return None

    degrees_of_freedom = first_count + second_count - 2

    first_variance = float(
        np.var(
            first,
            ddof=1,
        )
    )

    second_variance = float(
        np.var(
            second,
            ddof=1,
        )
    )

    pooled_variance = (
        ((first_count - 1) * first_variance) + ((second_count - 1) * second_variance)
    ) / degrees_of_freedom

    if not np.isfinite(pooled_variance) or pooled_variance <= 0.0:
        return None

    pooled_standard_deviation = float(np.sqrt(pooled_variance))

    cohen_d = (float(np.mean(first)) - float(np.mean(second))) / pooled_standard_deviation

    correction = 1.0 - 3.0 / (4.0 * degrees_of_freedom - 1.0)

    return float(correction * cohen_d)


def _rank_biserial(
    u_statistic: float,
    first_count: int,
    second_count: int,
) -> float | None:
    """Compute rank-biserial correlation from Mann-Whitney U."""

    denominator = first_count * second_count

    if denominator == 0:
        return None

    value = 2.0 * u_statistic / denominator - 1.0

    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


def _eta_squared(
    groups: Sequence[np.ndarray],
) -> float | None:
    """Compute eta-squared for a one-way ANOVA."""

    combined = np.concatenate(groups)

    overall_mean = float(np.mean(combined))

    between_sum_squares = sum(
        len(group) * (float(np.mean(group)) - overall_mean) ** 2 for group in groups
    )

    total_sum_squares = float(np.sum((combined - overall_mean) ** 2))

    if not np.isfinite(total_sum_squares) or total_sum_squares <= 0.0:
        return None

    return float(between_sum_squares / total_sum_squares)


def _epsilon_squared(
    statistic: float,
    *,
    total_count: int,
    group_count: int,
) -> float | None:
    """Compute epsilon-squared for Kruskal-Wallis."""

    denominator = total_count - group_count

    if denominator <= 0:
        return None

    value = (statistic - group_count + 1.0) / denominator

    return float(
        np.clip(
            value,
            0.0,
            1.0,
        )
    )


def compare_two_numeric_groups(
    data: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_a: object,
    group_b: object,
    *,
    method: TwoGroupMethod | str,
    alpha: float = 0.05,
) -> TwoGroupComparison:
    """
    Compare two explicitly selected independent numeric groups.

    Only rows belonging to the selected groups are candidates for the
    analysis. Missing and non-finite numeric values are excluded.
    """

    _require_distinct_columns(
        value_column,
        group_column,
    )

    value_series = _require_numeric_measure(
        data,
        value_column,
    )

    _require_column(
        data,
        group_column,
    )

    _validate_group_value(
        group_a,
        name="group_a",
    )

    _validate_group_value(
        group_b,
        name="group_b",
    )

    if group_a == group_b:
        raise ValueError("group_a and group_b must be different.")

    normalized_method = _coerce_two_group_method(method)

    normalized_alpha = _validate_alpha(alpha)

    numeric_values = _numeric_values(value_series)

    selected_mask = (
        data[group_column]
        .isin(
            [
                group_a,
                group_b,
            ]
        )
        .to_numpy(dtype=bool)
    )

    selected_row_count = int(np.sum(selected_mask))

    first = _extract_group_values(
        data,
        numeric_values=numeric_values,
        group_column=group_column,
        group_value=group_a,
    )

    second = _extract_group_values(
        data,
        numeric_values=numeric_values,
        group_column=group_column,
        group_value=group_b,
    )

    if normalized_method == TwoGroupMethod.WELCH_T:
        if len(first) < 2 or len(second) < 2:
            raise ValueError(
                "Welch's t-test requires at least two finite observations in each selected group."
            )

        test = ttest_ind(
            first,
            second,
            equal_var=False,
        )

        statistic = float(test.statistic)

        p_value = float(test.pvalue)

        effect_size_name = "hedges_g"

        effect_size = _hedges_g(
            first,
            second,
        )

    else:
        if len(first) < 1 or len(second) < 1:
            raise ValueError(
                "Mann-Whitney U requires at least one finite observation in each selected group."
            )

        test = mannwhitneyu(
            first,
            second,
            alternative="two-sided",
            method="auto",
        )

        statistic = float(test.statistic)

        p_value = float(test.pvalue)

        effect_size_name = "rank_biserial_correlation"

        effect_size = _rank_biserial(
            statistic,
            len(first),
            len(second),
        )

    used_row_count = len(first) + len(second)

    return TwoGroupComparison(
        value_column=value_column,
        group_column=group_column,
        group_a=group_a,
        group_b=group_b,
        method=normalized_method,
        alpha=normalized_alpha,
        source_row_count=len(data),
        selected_row_count=selected_row_count,
        used_row_count=used_row_count,
        excluded_value_row_count=(selected_row_count - used_row_count),
        group_a_summary=_summarize_group(
            group_a,
            first,
        ),
        group_b_summary=_summarize_group(
            group_b,
            second,
        ),
        statistic=statistic,
        p_value=p_value,
        significant=_safe_significance(
            p_value,
            normalized_alpha,
        ),
        effect_size_name=effect_size_name,
        effect_size=effect_size,
    )


def compare_multiple_numeric_groups(
    data: pd.DataFrame,
    value_column: str,
    group_column: str,
    group_values: Sequence[object],
    *,
    method: MultiGroupMethod | str,
    alpha: float = 0.05,
) -> MultiGroupComparison:
    """
    Compare multiple explicitly selected independent numeric groups.

    Missing and non-finite numeric observations from the selected groups
    are excluded. Rows from unselected groups are not part of the analysis.
    """

    _require_distinct_columns(
        value_column,
        group_column,
    )

    value_series = _require_numeric_measure(
        data,
        value_column,
    )

    _require_column(
        data,
        group_column,
    )

    normalized_groups = _validate_group_values(group_values)

    normalized_method = _coerce_multi_group_method(method)

    normalized_alpha = _validate_alpha(alpha)

    numeric_values = _numeric_values(value_series)

    selected_mask = data[group_column].isin(normalized_groups).to_numpy(dtype=bool)

    selected_row_count = int(np.sum(selected_mask))

    arrays = tuple(
        _extract_group_values(
            data,
            numeric_values=numeric_values,
            group_column=group_column,
            group_value=group,
        )
        for group in normalized_groups
    )

    if normalized_method == MultiGroupMethod.ONE_WAY_ANOVA:
        if any(len(values) < 2 for values in arrays):
            raise ValueError(
                "One-way ANOVA requires at least two finite observations in every selected group."
            )

        test = f_oneway(*arrays)

        statistic = float(test.statistic)

        p_value = float(test.pvalue)

        effect_size_name = "eta_squared"

        effect_size = _eta_squared(arrays)

    else:
        if any(len(values) < 1 for values in arrays):
            raise ValueError(
                "Kruskal-Wallis requires at least one finite observation in every selected group."
            )

        try:
            test = kruskal(*arrays)
        except ValueError as exc:
            raise ValueError(
                "Kruskal-Wallis could not be computed for the selected observations."
            ) from exc

        statistic = float(test.statistic)

        p_value = float(test.pvalue)

        effect_size_name = "epsilon_squared"

        effect_size = _epsilon_squared(
            statistic,
            total_count=sum(len(values) for values in arrays),
            group_count=len(arrays),
        )

    used_row_count = sum(len(values) for values in arrays)

    summaries = tuple(
        _summarize_group(
            group,
            values,
        )
        for group, values in zip(
            normalized_groups,
            arrays,
            strict=True,
        )
    )

    return MultiGroupComparison(
        value_column=value_column,
        group_column=group_column,
        groups=normalized_groups,
        method=normalized_method,
        alpha=normalized_alpha,
        source_row_count=len(data),
        selected_row_count=selected_row_count,
        used_row_count=used_row_count,
        excluded_value_row_count=(selected_row_count - used_row_count),
        group_summaries=summaries,
        statistic=statistic,
        p_value=p_value,
        significant=_safe_significance(
            p_value,
            normalized_alpha,
        ),
        effect_size_name=effect_size_name,
        effect_size=effect_size,
    )


def analyze_categorical_association(
    data: pd.DataFrame,
    row_column: str,
    column_column: str,
    *,
    alpha: float = 0.05,
    max_cells: int = 400,
) -> CategoricalAssociation:
    """
    Analyze association between two categorical variables.

    Rows missing either selected category are excluded. Pearson's
    chi-square test is computed without Yates continuity correction.
    """

    _require_distinct_columns(
        row_column,
        column_column,
    )

    row_series = _require_column(
        data,
        row_column,
    )

    column_series = _require_column(
        data,
        column_column,
    )

    normalized_alpha = _validate_alpha(alpha)

    normalized_max_cells = _validate_max_cells(max_cells)

    valid_mask = row_series.notna() & column_series.notna()

    included = data.loc[
        valid_mask,
        [
            row_column,
            column_column,
        ],
    ].copy()

    if included.empty:
        raise ValueError("No complete categorical observations are available.")

    row_level_count = int(included[row_column].nunique(dropna=True))

    column_level_count = int(included[column_column].nunique(dropna=True))

    if row_level_count < 2 or column_level_count < 2:
        raise ValueError(
            "Chi-square association requires at least two observed levels in each selected column."
        )

    cell_count = row_level_count * column_level_count

    if cell_count > normalized_max_cells:
        raise ValueError(
            "Contingency table is too large: "
            f"{cell_count:,} potential cells exceeds the "
            f"limit of {normalized_max_cells:,}."
        )

    observed = pd.crosstab(
        included[row_column],
        included[column_column],
        dropna=False,
    )

    test = chi2_contingency(
        observed.to_numpy(),
        correction=False,
    )

    statistic = float(test.statistic)

    p_value = float(test.pvalue)

    expected_values = np.asarray(
        test.expected_freq,
        dtype=float,
    )

    expected = pd.DataFrame(
        expected_values,
        index=observed.index.copy(),
        columns=observed.columns.copy(),
    )

    denominator_dimension = min(
        observed.shape[0] - 1,
        observed.shape[1] - 1,
    )

    if len(included) > 0 and denominator_dimension > 0:
        cramers_v = float(np.sqrt(statistic / (len(included) * denominator_dimension)))
    else:
        cramers_v = None

    below_five = expected_values < 5.0

    expected_below_five_count = int(np.sum(below_five))

    expected_below_five_ratio = float(expected_below_five_count / expected_values.size)

    return CategoricalAssociation(
        row_column=row_column,
        column_column=column_column,
        alpha=normalized_alpha,
        source_row_count=len(data),
        used_row_count=len(included),
        excluded_missing_row_count=(len(data) - len(included)),
        row_level_count=row_level_count,
        column_level_count=column_level_count,
        cell_count=cell_count,
        statistic=statistic,
        degrees_of_freedom=int(test.dof),
        p_value=p_value,
        significant=_safe_significance(
            p_value,
            normalized_alpha,
        ),
        cramers_v=cramers_v,
        minimum_expected_frequency=float(np.min(expected_values)),
        expected_below_five_count=expected_below_five_count,
        expected_below_five_ratio=expected_below_five_ratio,
        observed=observed.copy(),
        expected=expected,
    )
