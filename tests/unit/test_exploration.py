import numpy as np
import pandas as pd
import pytest

from data_diagnostics.analysis import (
    prepare_numeric_pair,
    summarize_categorical_frequencies,
    summarize_missingness,
    summarize_numeric_distribution,
    summarize_numeric_outliers,
)


def test_summarize_missingness_preserves_column_order() -> None:
    data = pd.DataFrame(
        {
            "a": [1, None, 3, None],
            "b": [1, 2, 3, 4],
        }
    )

    result = summarize_missingness(data)

    assert [summary.column for summary in result] == [
        "a",
        "b",
    ]

    assert result[0].missing_count == 2
    assert result[0].missing_ratio == pytest.approx(0.5)

    assert result[1].missing_count == 0
    assert result[1].missing_ratio == pytest.approx(0.0)


def test_numeric_distribution_excludes_missing_and_infinite_values() -> None:
    data = pd.DataFrame(
        {
            "value": [
                1.0,
                2.0,
                3.0,
                np.nan,
                np.inf,
                -np.inf,
            ]
        }
    )

    result = summarize_numeric_distribution(
        data,
        "value",
        bins=3,
    )

    assert result.total_count == 6
    assert result.valid_count == 3
    assert result.missing_count == 1
    assert result.non_finite_count == 2

    assert result.mean == pytest.approx(2.0)
    assert result.median == pytest.approx(2.0)
    assert result.minimum == pytest.approx(1.0)
    assert result.maximum == pytest.approx(3.0)

    assert len(result.histogram) == 3

    assert sum(histogram_bin.count for histogram_bin in result.histogram) == 3


def test_numeric_distribution_handles_no_finite_values() -> None:
    data = pd.DataFrame(
        {
            "value": [
                np.nan,
                np.inf,
                -np.inf,
            ]
        }
    )

    result = summarize_numeric_distribution(
        data,
        "value",
    )

    assert result.valid_count == 0
    assert result.mean is None
    assert result.median is None
    assert result.histogram == ()


def test_numeric_distribution_rejects_non_numeric_column() -> None:
    data = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Mumbai",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        summarize_numeric_distribution(
            data,
            "city",
        )


def test_numeric_distribution_rejects_invalid_bin_count() -> None:
    data = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(ValueError):
        summarize_numeric_distribution(
            data,
            "value",
            bins=0,
        )


def test_numeric_outliers_detect_iqr_outlier() -> None:
    data = pd.DataFrame(
        {
            "value": [
                1,
                2,
                2,
                3,
                3,
                4,
                100,
            ]
        }
    )

    result = summarize_numeric_outliers(
        data,
        "value",
    )

    assert result.valid_count == 7
    assert result.upper_outlier_count == 1
    assert result.lower_outlier_count == 0
    assert result.outlier_count == 1
    assert result.outlier_ratio == pytest.approx(1 / 7)


def test_numeric_outliers_do_not_modify_source_data() -> None:
    data = pd.DataFrame(
        {
            "value": [
                1.0,
                2.0,
                100.0,
                np.nan,
            ]
        }
    )

    original = data.copy(deep=True)

    summarize_numeric_outliers(
        data,
        "value",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_categorical_frequencies_include_other_and_missing_counts() -> None:
    data = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Delhi",
                "Delhi",
                "Mumbai",
                "Mumbai",
                "Bengaluru",
                "Pune",
                None,
            ]
        }
    )

    result = summarize_categorical_frequencies(
        data,
        "city",
        top_n=2,
    )

    assert result.total_count == 8
    assert result.non_missing_count == 7
    assert result.missing_count == 1
    assert result.unique_count == 4

    assert result.frequencies[0].value == "Delhi"
    assert result.frequencies[0].count == 3

    assert result.frequencies[1].value == "Mumbai"
    assert result.frequencies[1].count == 2

    assert result.other_count == 2
    assert result.other_ratio == pytest.approx(2 / 8)


def test_categorical_frequencies_reject_invalid_top_n() -> None:
    data = pd.DataFrame(
        {
            "city": ["Delhi", "Mumbai"],
        }
    )

    with pytest.raises(ValueError):
        summarize_categorical_frequencies(
            data,
            "city",
            top_n=0,
        )


def test_prepare_numeric_pair_excludes_incomplete_and_non_finite_rows() -> None:
    data = pd.DataFrame(
        {
            "x": [
                1.0,
                2.0,
                np.nan,
                4.0,
                np.inf,
            ],
            "y": [
                10.0,
                np.nan,
                30.0,
                40.0,
                50.0,
            ],
        }
    )

    result = prepare_numeric_pair(
        data,
        "x",
        "y",
    )

    assert result.total_count == 5
    assert result.valid_count == 2
    assert result.excluded_count == 3

    expected = pd.DataFrame(
        {
            "x": [1.0, 4.0],
            "y": [10.0, 40.0],
        },
        index=[0, 3],
    )

    pd.testing.assert_frame_equal(
        result.data,
        expected,
    )


def test_prepare_numeric_pair_requires_distinct_columns() -> None:
    data = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        prepare_numeric_pair(
            data,
            "value",
            "value",
        )


def test_exploration_unknown_column_raises_clear_error() -> None:
    data = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(
        ValueError,
        match="Unknown column",
    ):
        summarize_numeric_distribution(
            data,
            "missing",
        )


def test_numeric_outliers_reject_boolean_multiplier() -> None:
    data = pd.DataFrame(
        {
            "value": [1, 2, 3],
        }
    )

    with pytest.raises(
        TypeError,
        match="positive finite number",
    ):
        summarize_numeric_outliers(
            data,
            "value",
            multiplier=True,
        )
