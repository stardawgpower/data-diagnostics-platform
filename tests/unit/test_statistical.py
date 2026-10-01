import numpy as np
import pandas as pd
import pytest

from data_diagnostics.analysis.statistical import (
    MultiGroupMethod,
    TwoGroupMethod,
    analyze_categorical_association,
    compare_multiple_numeric_groups,
    compare_two_numeric_groups,
)


def test_welch_t_comparison_returns_group_summaries() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
        }
    )

    result = compare_two_numeric_groups(
        data,
        "value",
        "group",
        "A",
        "B",
        method="welch_t",
    )

    assert result.method == TwoGroupMethod.WELCH_T
    assert result.group_a_summary.count == 3
    assert result.group_b_summary.count == 3
    assert result.group_a_summary.mean == pytest.approx(2.0)
    assert result.group_b_summary.mean == pytest.approx(6.0)
    assert result.statistic < 0
    assert 0.0 <= result.p_value <= 1.0
    assert result.effect_size_name == "hedges_g"
    assert result.effect_size is not None
    assert result.effect_size < 0


def test_mann_whitney_returns_rank_biserial_effect_size() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        }
    )

    result = compare_two_numeric_groups(
        data,
        "value",
        "group",
        "A",
        "B",
        method="mann_whitney",
    )

    assert result.method == TwoGroupMethod.MANN_WHITNEY
    assert result.effect_size_name == "rank_biserial_correlation"
    assert result.effect_size == pytest.approx(-1.0)


def test_two_group_analysis_can_select_two_groups_from_many() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0],
        }
    )

    result = compare_two_numeric_groups(
        data,
        "value",
        "group",
        "A",
        "B",
        method="mann_whitney",
    )

    assert result.source_row_count == 6
    assert result.selected_row_count == 4
    assert result.used_row_count == 4
    assert result.excluded_value_row_count == 0


def test_two_group_analysis_excludes_missing_and_nonfinite_values() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "value": [1.0, np.nan, np.inf, 4.0, 5.0, -np.inf],
        }
    )

    result = compare_two_numeric_groups(
        data,
        "value",
        "group",
        "A",
        "B",
        method="mann_whitney",
    )

    assert result.selected_row_count == 6
    assert result.used_row_count == 3
    assert result.excluded_value_row_count == 3
    assert result.group_a_summary.count == 1
    assert result.group_b_summary.count == 2


def test_two_group_rejects_non_numeric_measure() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "value": ["x", "y"],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        compare_two_numeric_groups(
            data,
            "value",
            "group",
            "A",
            "B",
            method="mann_whitney",
        )


def test_two_group_rejects_boolean_measure() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "value": [True, False],
        }
    )

    with pytest.raises(
        ValueError,
        match="non-boolean",
    ):
        compare_two_numeric_groups(
            data,
            "value",
            "group",
            "A",
            "B",
            method="mann_whitney",
        )


def test_two_group_requires_distinct_groups() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        compare_two_numeric_groups(
            data,
            "value",
            "group",
            "A",
            "A",
            method="mann_whitney",
        )


def test_welch_t_requires_two_finite_values_per_group() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "B", "B"],
            "value": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two finite",
    ):
        compare_two_numeric_groups(
            data,
            "value",
            "group",
            "A",
            "B",
            method="welch_t",
        )


def test_two_group_analysis_does_not_modify_source() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )

    original = data.copy(deep=True)

    compare_two_numeric_groups(
        data,
        "value",
        "group",
        "A",
        "B",
        method="mann_whitney",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_one_way_anova_returns_eta_squared() -> None:
    data = pd.DataFrame(
        {
            "group": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
            ],
            "value": [
                1.0,
                2.0,
                3.0,
                5.0,
                6.0,
                7.0,
                9.0,
                10.0,
                11.0,
            ],
        }
    )

    result = compare_multiple_numeric_groups(
        data,
        "value",
        "group",
        ["A", "B", "C"],
        method="one_way_anova",
    )

    assert result.method == MultiGroupMethod.ONE_WAY_ANOVA
    assert len(result.group_summaries) == 3
    assert result.used_row_count == 9
    assert result.effect_size_name == "eta_squared"
    assert result.effect_size is not None
    assert 0.0 <= result.effect_size <= 1.0


def test_kruskal_wallis_returns_epsilon_squared() -> None:
    data = pd.DataFrame(
        {
            "group": [
                "A",
                "A",
                "B",
                "B",
                "C",
                "C",
            ],
            "value": [
                1.0,
                2.0,
                4.0,
                5.0,
                8.0,
                9.0,
            ],
        }
    )

    result = compare_multiple_numeric_groups(
        data,
        "value",
        "group",
        ["A", "B", "C"],
        method="kruskal_wallis",
    )

    assert result.method == MultiGroupMethod.KRUSKAL_WALLIS
    assert result.effect_size_name == "epsilon_squared"
    assert result.effect_size is not None
    assert 0.0 <= result.effect_size <= 1.0


def test_multi_group_analysis_uses_only_selected_groups() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 100.0, 200.0],
        }
    )

    result = compare_multiple_numeric_groups(
        data,
        "value",
        "group",
        ["A", "B", "C"],
        method="kruskal_wallis",
    )

    assert result.source_row_count == 8
    assert result.selected_row_count == 6
    assert result.used_row_count == 6
    assert result.groups == ("A", "B", "C")


def test_multi_group_rejects_duplicate_group_values() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="must not contain duplicates",
    ):
        compare_multiple_numeric_groups(
            data,
            "value",
            "group",
            ["A", "A"],
            method="kruskal_wallis",
        )


def test_multi_group_requires_at_least_two_groups() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="At least two group values",
    ):
        compare_multiple_numeric_groups(
            data,
            "value",
            "group",
            ["A"],
            method="kruskal_wallis",
        )


def test_multi_group_excludes_nonfinite_values() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "C", "C"],
            "value": [1.0, np.nan, 2.0, np.inf, 3.0, 4.0],
        }
    )

    result = compare_multiple_numeric_groups(
        data,
        "value",
        "group",
        ["A", "B", "C"],
        method="kruskal_wallis",
    )

    assert result.selected_row_count == 6
    assert result.used_row_count == 4
    assert result.excluded_value_row_count == 2


def test_anova_requires_two_finite_observations_per_group() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "A", "B"],
            "value": [1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two finite",
    ):
        compare_multiple_numeric_groups(
            data,
            "value",
            "group",
            ["A", "B"],
            method="one_way_anova",
        )


def test_categorical_association_builds_expected_table() -> None:
    data = pd.DataFrame(
        {
            "segment": ["A", "A", "A", "B", "B", "B", "B", "A"],
            "outcome": ["Yes", "Yes", "No", "Yes", "No", "No", "No", "Yes"],
        }
    )

    result = analyze_categorical_association(
        data,
        "segment",
        "outcome",
    )

    assert result.row_level_count == 2
    assert result.column_level_count == 2
    assert result.cell_count == 4
    assert result.used_row_count == 8
    assert result.degrees_of_freedom == 1
    assert result.observed.shape == (2, 2)
    assert result.expected.shape == (2, 2)
    assert result.cramers_v is not None
    assert 0.0 <= result.cramers_v <= 1.0


def test_categorical_association_excludes_missing_rows() -> None:
    data = pd.DataFrame(
        {
            "segment": ["A", "A", "B", "B", None],
            "outcome": ["Yes", "No", "Yes", "No", "Yes"],
        }
    )

    result = analyze_categorical_association(
        data,
        "segment",
        "outcome",
    )

    assert result.source_row_count == 5
    assert result.used_row_count == 4
    assert result.excluded_missing_row_count == 1


def test_categorical_association_has_cell_guard() -> None:
    data = pd.DataFrame(
        {
            "left": ["A", "A", "B", "B"],
            "right": ["X", "Y", "X", "Y"],
        }
    )

    with pytest.raises(
        ValueError,
        match="too large",
    ):
        analyze_categorical_association(
            data,
            "left",
            "right",
            max_cells=3,
        )


def test_categorical_association_requires_distinct_columns() -> None:
    data = pd.DataFrame(
        {
            "category": ["A", "A", "B", "B"],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be distinct",
    ):
        analyze_categorical_association(
            data,
            "category",
            "category",
        )


def test_categorical_association_requires_two_levels_each() -> None:
    data = pd.DataFrame(
        {
            "left": ["A", "A", "A"],
            "right": ["X", "Y", "X"],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least two",
    ):
        analyze_categorical_association(
            data,
            "left",
            "right",
        )


def test_invalid_alpha_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "group": ["A", "B"],
            "value": [1.0, 2.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        compare_two_numeric_groups(
            data,
            "value",
            "group",
            "A",
            "B",
            method="mann_whitney",
            alpha=1.0,
        )


def test_categorical_association_does_not_modify_source() -> None:
    data = pd.DataFrame(
        {
            "left": ["A", "A", "B", "B"],
            "right": ["X", "Y", "X", "Y"],
        }
    )

    original = data.copy(deep=True)

    analyze_categorical_association(
        data,
        "left",
        "right",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )
