import pandas as pd
import pytest

from data_diagnostics.quality import (
    QualityIssueCode,
    analyze_data_quality,
)


def _issue_codes(report):
    return {issue.code for issue in report.issues}


def _issues_for_column(report, column_name):
    return [issue for issue in report.issues if issue.column == column_name]


def test_quality_report_counts_missing_cells() -> None:
    data = pd.DataFrame(
        {
            "a": [1, None, 3, None],
            "b": ["x", "y", None, "z"],
        }
    )

    report = analyze_data_quality(data)

    assert report.row_count == 4
    assert report.column_count == 2
    assert report.missing_cell_count == 3
    assert report.missing_cell_ratio == pytest.approx(3 / 8)


def test_detect_duplicate_rows() -> None:
    data = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Alice"],
            "score": [10, 20, 10],
        }
    )

    report = analyze_data_quality(data)

    assert report.duplicate_row_count == 1
    assert report.duplicate_row_ratio == pytest.approx(1 / 3)
    assert QualityIssueCode.DUPLICATE_ROWS in _issue_codes(report)


def test_detect_all_missing_column() -> None:
    data = pd.DataFrame(
        {
            "empty_column": [None, None, None],
            "value": [1, 2, 3],
        }
    )

    report = analyze_data_quality(data)
    issues = _issues_for_column(report, "empty_column")

    assert any(issue.code == QualityIssueCode.ALL_MISSING for issue in issues)


def test_detect_constant_column() -> None:
    data = pd.DataFrame(
        {
            "country": ["India", "India", "India", "India"],
        }
    )

    report = analyze_data_quality(data)

    assert QualityIssueCode.CONSTANT in _issue_codes(report)


def test_detect_high_missingness() -> None:
    data = pd.DataFrame(
        {
            "value": [10, None, None, None, 50],
        }
    )

    report = analyze_data_quality(
        data,
        high_missingness_threshold=0.50,
    )

    issues = _issues_for_column(report, "value")

    assert any(issue.code == QualityIssueCode.HIGH_MISSINGNESS for issue in issues)


def test_detect_probable_identifier() -> None:
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C005",
            ]
        }
    )

    report = analyze_data_quality(data)
    issues = _issues_for_column(report, "customer_id")

    assert any(issue.code == QualityIssueCode.PROBABLE_IDENTIFIER for issue in issues)


def test_detect_high_cardinality_text() -> None:
    data = pd.DataFrame({"description": [f"unique description {index}" for index in range(30)]})

    report = analyze_data_quality(
        data,
        high_cardinality_min_unique=20,
        high_cardinality_ratio=0.80,
    )

    issues = _issues_for_column(report, "description")

    assert any(issue.code == QualityIssueCode.HIGH_CARDINALITY for issue in issues)


def test_quality_analysis_does_not_modify_dataframe() -> None:
    data = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "value": [10, None, 30],
        }
    )

    original = data.copy(deep=True)

    analyze_data_quality(data)

    pd.testing.assert_frame_equal(data, original)


@pytest.mark.parametrize(
    ("argument", "value"),
    [
        ("high_missingness_threshold", -0.1),
        ("high_missingness_threshold", 1.1),
        ("high_cardinality_ratio", -0.1),
        ("high_cardinality_ratio", 1.1),
        ("high_cardinality_min_unique", 0),
    ],
)
def test_invalid_quality_configuration(argument, value) -> None:
    data = pd.DataFrame({"value": [1, 2, 3]})

    kwargs = {argument: value}

    with pytest.raises(ValueError):
        analyze_data_quality(data, **kwargs)
