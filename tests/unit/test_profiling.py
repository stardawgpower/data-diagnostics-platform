import pandas as pd
import pytest

from data_diagnostics.analysis import profile_dataset
from data_diagnostics.ingestion import SemanticType
from data_diagnostics.quality import QualityIssueCode


def _profiles_by_name(profile):
    """Return column profiles indexed by column name."""

    return {column.schema.name: column for column in profile.columns}


def test_profile_dataset_dimensions() -> None:
    data = pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Charlie"],
            "score": [80, 90, 100],
        }
    )

    profile = profile_dataset(data)

    assert profile.row_count == 3
    assert profile.column_count == 2
    assert len(profile.columns) == 2


def test_profile_numeric_column() -> None:
    data = pd.DataFrame(
        {
            "revenue": [10.5, 20.25, 30.75, None],
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["revenue"]

    assert column.schema.semantic_type == SemanticType.NUMERIC_CONTINUOUS
    assert column.numeric is not None

    assert column.numeric.count == 3
    assert column.numeric.mean == pytest.approx(20.5)
    assert column.numeric.median == pytest.approx(20.25)
    assert column.numeric.minimum == pytest.approx(10.5)
    assert column.numeric.maximum == pytest.approx(30.75)

    expected_std = pd.Series([10.5, 20.25, 30.75]).std()

    assert column.numeric.standard_deviation == pytest.approx(expected_std)


def test_profile_categorical_column() -> None:
    data = pd.DataFrame(
        {
            "city": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Bengaluru",
                "Delhi",
            ]
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["city"]

    assert column.schema.semantic_type == SemanticType.CATEGORICAL
    assert column.categorical is not None

    assert column.categorical.count == 5
    assert column.categorical.unique_count == 3
    assert column.categorical.most_frequent_value == "Delhi"
    assert column.categorical.most_frequent_count == 3


def test_profile_binary_column_as_categorical_summary() -> None:
    data = pd.DataFrame(
        {
            "converted": [0, 1, 1, 0, 1],
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["converted"]

    assert column.schema.semantic_type == SemanticType.BINARY
    assert column.categorical is not None

    assert column.categorical.unique_count == 2
    assert column.categorical.most_frequent_value == 1
    assert column.categorical.most_frequent_count == 3


def test_profile_native_datetime_column() -> None:
    data = pd.DataFrame(
        {
            "event_time": pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                ]
            )
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["event_time"]

    assert column.schema.semantic_type == SemanticType.DATETIME
    assert column.datetime is not None

    assert column.datetime.count == 3
    assert column.datetime.earliest == pd.Timestamp("2026-01-01")
    assert column.datetime.latest == pd.Timestamp("2026-01-03")


def test_profile_datetime_string_column() -> None:
    data = pd.DataFrame(
        {
            "order_date": [
                "2026-02-03",
                "2026-02-01",
                "2026-02-02",
            ]
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["order_date"]

    assert column.schema.semantic_type == SemanticType.DATETIME
    assert column.datetime is not None

    assert column.datetime.count == 3
    assert column.datetime.earliest == pd.Timestamp("2026-02-01")
    assert column.datetime.latest == pd.Timestamp("2026-02-03")


def test_profile_day_first_datetime_uses_inferred_format() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "10/03/2004",
                "13/03/2004",
                "04/04/2005",
            ]
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["Date"]

    assert column.schema.temporal_format == "%d/%m/%Y"
    assert column.datetime is not None
    assert column.datetime.earliest == pd.Timestamp("2004-03-10")
    assert column.datetime.latest == pd.Timestamp("2005-04-04")


def test_profile_time_only_column() -> None:
    data = pd.DataFrame(
        {
            "Time": [
                "18.00.00",
                "01.00.00",
                "23.00.00",
            ]
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["Time"]

    assert column.schema.semantic_type == SemanticType.TIME
    assert column.time is not None
    assert column.time.count == 3
    assert column.time.earliest.isoformat() == "01:00:00"
    assert column.time.latest.isoformat() == "23:00:00"


def test_profile_ambiguous_datetime_does_not_guess_range() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "01/02/2026",
                "02/03/2026",
                "03/04/2026",
            ]
        }
    )

    profile = profile_dataset(data)
    column = _profiles_by_name(profile)["Date"]

    assert column.schema.semantic_type == SemanticType.DATETIME
    assert column.schema.temporal_format is None
    assert column.datetime is not None
    assert column.datetime.earliest is None
    assert column.datetime.latest is None


def test_profile_includes_quality_report() -> None:
    data = pd.DataFrame(
        {
            "country": ["India", "India", "India"],
            "value": [10, 20, 30],
        }
    )

    profile = profile_dataset(data)

    issue_codes = {issue.code for issue in profile.quality.issues}

    assert QualityIssueCode.CONSTANT in issue_codes


def test_profile_all_missing_column_has_no_descriptive_summary() -> None:
    data = pd.DataFrame(
        {
            "empty": [None, None, None],
        }
    )

    profile = profile_dataset(data)
    column = profile.columns[0]

    assert column.schema.semantic_type == SemanticType.UNKNOWN
    assert column.numeric is None
    assert column.categorical is None
    assert column.datetime is None
    assert column.time is None


def test_profile_does_not_modify_dataframe() -> None:
    data = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "order_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
            "value": [10.5, 20.5, None],
        }
    )

    original = data.copy(deep=True)

    profile_dataset(data)

    pd.testing.assert_frame_equal(data, original)
