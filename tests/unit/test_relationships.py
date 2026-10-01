import numpy as np
import pandas as pd
import pytest

from data_diagnostics.analysis.relationships import (
    CorrelationMethod,
    RelationshipStatus,
    analyze_numeric_relationship,
    build_correlation_matrix,
    strongest_correlations,
)


def test_pearson_relationship_detects_perfect_linear_association() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [2, 4, 6, 8, 10],
        }
    )

    result = analyze_numeric_relationship(
        data,
        "x",
        "y",
    )

    assert result.method == CorrelationMethod.PEARSON
    assert result.status == RelationshipStatus.OK
    assert result.valid_count == 5
    assert result.excluded_count == 0
    assert result.coefficient == pytest.approx(1.0)


def test_spearman_relationship_detects_monotonic_association() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [1, 4, 9, 16, 25],
        }
    )

    pearson = analyze_numeric_relationship(
        data,
        "x",
        "y",
        method="pearson",
    )

    spearman = analyze_numeric_relationship(
        data,
        "x",
        "y",
        method="spearman",
    )

    assert pearson.coefficient is not None
    assert spearman.coefficient == pytest.approx(1.0)
    assert pearson.coefficient < spearman.coefficient


def test_relationship_excludes_missing_and_non_finite_pairs() -> None:
    data = pd.DataFrame(
        {
            "x": [
                1.0,
                2.0,
                np.nan,
                4.0,
                np.inf,
                6.0,
            ],
            "y": [
                10.0,
                20.0,
                30.0,
                np.nan,
                50.0,
                60.0,
            ],
        }
    )

    result = analyze_numeric_relationship(
        data,
        "x",
        "y",
    )

    assert result.total_count == 6
    assert result.valid_count == 3
    assert result.excluded_count == 3
    assert result.coefficient == pytest.approx(1.0)


def test_relationship_reports_insufficient_pairs() -> None:
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan],
            "y": [10.0, np.nan, 30.0],
        }
    )

    result = analyze_numeric_relationship(
        data,
        "x",
        "y",
        minimum_pairs=2,
    )

    assert result.valid_count == 1
    assert result.coefficient is None
    assert result.status == RelationshipStatus.INSUFFICIENT_PAIRS


def test_relationship_reports_constant_input() -> None:
    data = pd.DataFrame(
        {
            "x": [5.0, 5.0, 5.0, 5.0],
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = analyze_numeric_relationship(
        data,
        "x",
        "y",
    )

    assert result.coefficient is None
    assert result.status == RelationshipStatus.CONSTANT_INPUT


def test_relationship_requires_distinct_columns() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        analyze_numeric_relationship(
            data,
            "x",
            "x",
        )


def test_relationship_rejects_non_numeric_column() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "city": [
                "Delhi",
                "Mumbai",
                "Pune",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        analyze_numeric_relationship(
            data,
            "x",
            "city",
        )


def test_relationship_rejects_unknown_column() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )

    with pytest.raises(
        ValueError,
        match="Unknown column",
    ):
        analyze_numeric_relationship(
            data,
            "x",
            "missing",
        )


def test_relationship_does_not_modify_source_data() -> None:
    data = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [10.0, 20.0, 30.0, 40.0],
        }
    )

    original = data.copy(
        deep=True,
    )

    analyze_numeric_relationship(
        data,
        "x",
        "y",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_correlation_matrix_is_symmetric() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],
            "c": [5, 4, 3, 2, 1],
        }
    )

    result = build_correlation_matrix(
        data,
        ["a", "b", "c"],
    )

    pd.testing.assert_frame_equal(
        result.coefficients,
        result.coefficients.T,
    )

    assert result.coefficients.loc[
        "a",
        "a",
    ] == pytest.approx(1.0)

    assert result.coefficients.loc[
        "a",
        "b",
    ] == pytest.approx(1.0)

    assert result.coefficients.loc[
        "a",
        "c",
    ] == pytest.approx(-1.0)


def test_correlation_matrix_records_pairwise_valid_counts() -> None:
    data = pd.DataFrame(
        {
            "a": [
                1.0,
                2.0,
                3.0,
                np.nan,
                5.0,
            ],
            "b": [
                2.0,
                4.0,
                np.nan,
                8.0,
                10.0,
            ],
            "c": [
                5.0,
                4.0,
                3.0,
                2.0,
                1.0,
            ],
        }
    )

    result = build_correlation_matrix(
        data,
        ["a", "b", "c"],
    )

    assert (
        result.valid_counts.loc[
            "a",
            "a",
        ]
        == 4
    )

    assert (
        result.valid_counts.loc[
            "a",
            "b",
        ]
        == 3
    )

    assert (
        result.valid_counts.loc[
            "a",
            "c",
        ]
        == 4
    )

    assert (
        result.valid_counts.loc[
            "b",
            "c",
        ]
        == 4
    )


def test_correlation_matrix_leaves_constant_relationship_undefined() -> None:
    data = pd.DataFrame(
        {
            "constant": [5, 5, 5, 5],
            "value": [1, 2, 3, 4],
        }
    )

    result = build_correlation_matrix(
        data,
        [
            "constant",
            "value",
        ],
    )

    assert pd.isna(
        result.coefficients.loc[
            "constant",
            "constant",
        ]
    )

    assert pd.isna(
        result.coefficients.loc[
            "constant",
            "value",
        ]
    )

    assert result.coefficients.loc[
        "value",
        "value",
    ] == pytest.approx(1.0)


def test_strongest_correlations_orders_by_absolute_magnitude() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [2, 4, 6, 8, 10, 12],
            "c": [6, 5, 4, 3, 2, 1],
            "d": [1, 1, 2, 2, 3, 4],
        }
    )

    matrix = build_correlation_matrix(
        data,
        ["a", "b", "c", "d"],
    )

    relationships = strongest_correlations(
        matrix,
        top_n=3,
    )

    assert len(relationships) == 3

    absolute_values = [relationship.absolute_coefficient for relationship in relationships]

    assert absolute_values == sorted(
        absolute_values,
        reverse=True,
    )


def test_strongest_correlations_applies_absolute_threshold() -> None:
    data = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [2, 4, 6, 8, 10, 12],
            "c": [3, 1, 6, 2, 5, 4],
        }
    )

    matrix = build_correlation_matrix(
        data,
        ["a", "b", "c"],
    )

    relationships = strongest_correlations(
        matrix,
        minimum_absolute=0.95,
    )

    assert relationships

    assert all(relationship.absolute_coefficient >= 0.95 for relationship in relationships)


def test_minimum_pairs_validation() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )

    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        analyze_numeric_relationship(
            data,
            "x",
            "y",
            minimum_pairs=1,
        )

    with pytest.raises(
        TypeError,
        match="integer",
    ):
        analyze_numeric_relationship(
            data,
            "x",
            "y",
            minimum_pairs=True,
        )


def test_correlation_matrix_rejects_duplicate_columns() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        build_correlation_matrix(
            data,
            [
                "x",
                "x",
            ],
        )


def test_strongest_correlations_validates_top_n() -> None:
    data = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
        }
    )

    matrix = build_correlation_matrix(
        data,
        ["x", "y"],
    )

    with pytest.raises(
        ValueError,
        match="at least 1",
    ):
        strongest_correlations(
            matrix,
            top_n=0,
        )
