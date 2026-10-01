import numpy as np
import pandas as pd
import pytest

from data_diagnostics.analysis.pivot import (
    PivotAggregation,
    build_pivot_table,
    estimate_pivot_cardinality,
)


def test_row_count_pivot_with_single_dimension() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                "North",
                "South",
            ]
        }
    )

    result = build_pivot_table(
        data,
        ["region"],
    )

    assert result.aggregation == PivotAggregation.ROW_COUNT
    assert result.source_row_count == 3
    assert result.included_row_count == 3
    assert result.group_count == 2

    assert (
        result.table.loc[
            "North",
            "value",
        ]
        == 2
    )

    assert (
        result.table.loc[
            "South",
            "value",
        ]
        == 1
    )


def test_row_and_column_dimensions_build_matrix() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                "North",
                "South",
            ],
            "channel": [
                "Online",
                "Store",
                "Online",
            ],
        }
    )

    result = build_pivot_table(
        data,
        ["region"],
        ["channel"],
    )

    assert result.cardinality.row_group_count == 2
    assert result.cardinality.column_group_count == 2
    assert result.cardinality.potential_cell_count == 4

    assert (
        result.table.loc[
            "North",
            "Online",
        ]
        == 1
    )

    assert (
        result.table.loc[
            "North",
            "Store",
        ]
        == 1
    )

    assert (
        result.table.loc[
            "South",
            "Online",
        ]
        == 1
    )

    assert pd.isna(
        result.table.loc[
            "South",
            "Store",
        ]
    )


def test_numeric_mean_aggregation() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                "North",
                "South",
            ],
            "sales": [
                10.0,
                20.0,
                30.0,
            ],
        }
    )

    result = build_pivot_table(
        data,
        ["region"],
        value_column="sales",
        aggregation="mean",
    )

    assert result.aggregation == PivotAggregation.MEAN

    assert result.table.loc[
        "North",
        "value",
    ] == pytest.approx(15.0)

    assert result.table.loc[
        "South",
        "value",
    ] == pytest.approx(30.0)


def test_count_aggregation_counts_non_missing_values() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                "North",
                "North",
            ],
            "sales": [
                10.0,
                np.nan,
                30.0,
            ],
        }
    )

    result = build_pivot_table(
        data,
        ["region"],
        value_column="sales",
        aggregation="count",
    )

    assert (
        result.table.loc[
            "North",
            "value",
        ]
        == 2
    )


def test_missing_dimensions_are_excluded_by_default() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                None,
                "South",
            ],
            "sales": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    result = build_pivot_table(
        data,
        ["region"],
        value_column="sales",
        aggregation="sum",
    )

    assert result.source_row_count == 3
    assert result.included_row_count == 2
    assert result.excluded_dimension_row_count == 1
    assert result.group_count == 2


def test_missing_dimensions_can_be_included() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                None,
                "South",
            ]
        }
    )

    result = build_pivot_table(
        data,
        ["region"],
        include_missing_dimensions=True,
    )

    assert result.included_row_count == 3
    assert result.excluded_dimension_row_count == 0
    assert result.group_count == 3


def test_numeric_aggregation_rejects_non_numeric_values() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
            ],
            "label": [
                "A",
                "B",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        build_pivot_table(
            data,
            ["region"],
            value_column="label",
            aggregation="mean",
        )


def test_numeric_aggregation_rejects_boolean_values() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
            ],
            "flag": [
                True,
                False,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        build_pivot_table(
            data,
            ["region"],
            value_column="flag",
            aggregation="sum",
        )


def test_unknown_dimension_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="Unknown column",
    ):
        build_pivot_table(
            data,
            ["missing"],
        )


def test_duplicate_dimensions_are_rejected() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="must not contain duplicate",
    ):
        build_pivot_table(
            data,
            [
                "region",
                "region",
            ],
        )


def test_row_and_column_dimension_overlap_is_rejected() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="must be distinct",
    ):
        build_pivot_table(
            data,
            ["region"],
            ["region"],
        )


def test_value_column_cannot_also_be_dimension() -> None:
    data = pd.DataFrame(
        {
            "sales": [
                1.0,
                2.0,
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="different from all pivot dimensions",
    ):
        build_pivot_table(
            data,
            ["sales"],
            value_column="sales",
            aggregation="mean",
        )


def test_at_least_one_row_dimension_is_required() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="At least one row dimension",
    ):
        build_pivot_table(
            data,
            [],
        )


def test_row_count_rejects_value_column() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
            ],
            "sales": [
                10.0,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be None",
    ):
        build_pivot_table(
            data,
            ["region"],
            value_column="sales",
            aggregation="row_count",
        )


def test_cardinality_limit_prevents_large_pivot() -> None:
    data = pd.DataFrame(
        {
            "row": [
                "A",
                "A",
                "B",
                "B",
            ],
            "column": [
                "X",
                "Y",
                "X",
                "Y",
            ],
        }
    )

    cardinality = estimate_pivot_cardinality(
        data,
        ["row"],
        ["column"],
        max_cells=3,
    )

    assert cardinality.potential_cell_count == 4

    with pytest.raises(
        ValueError,
        match="potential cells",
    ):
        build_pivot_table(
            data,
            ["row"],
            ["column"],
            max_cells=3,
        )


def test_pivot_analysis_does_not_modify_source_data() -> None:
    data = pd.DataFrame(
        {
            "region": [
                "North",
                "South",
            ],
            "channel": [
                "Online",
                "Store",
            ],
            "sales": [
                10.0,
                20.0,
            ],
        }
    )

    original = data.copy(deep=True)

    build_pivot_table(
        data,
        ["region"],
        ["channel"],
        value_column="sales",
        aggregation="sum",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )
