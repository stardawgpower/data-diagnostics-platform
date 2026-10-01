import numpy as np
import pandas as pd
import pytest

from data_diagnostics.analysis.time_analysis import (
    TemporalAggregation,
    TemporalFrequency,
    aggregate_temporal_numeric_series,
    analyze_temporal_sampling,
    detect_time_gaps,
    prepare_temporal_axis,
    prepare_temporal_numeric_series,
)


def test_prepare_temporal_axis_accepts_native_datetime() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 12:00:00",
                ]
            )
        }
    )

    result = prepare_temporal_axis(
        data,
        "timestamp",
    )

    assert result.total_count == 3
    assert result.valid_count == 3
    assert result.invalid_count == 0

    pd.testing.assert_series_equal(
        result.timestamps,
        pd.Series(
            pd.to_datetime(
                [
                    "2026-01-01 10:00:00",
                    "2026-01-01 11:00:00",
                    "2026-01-01 12:00:00",
                ]
            ),
            name="timestamp",
        ),
    )


def test_prepare_temporal_axis_uses_explicit_date_format() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "10/03/2004",
                "11/03/2004",
            ]
        }
    )

    result = prepare_temporal_axis(
        data,
        "Date",
        date_format="%d/%m/%Y",
    )

    assert result.timestamps.iloc[0] == pd.Timestamp("2004-03-10")

    assert result.timestamps.iloc[1] == pd.Timestamp("2004-03-11")


def test_prepare_temporal_axis_requires_format_for_text_dates() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "01/02/2026",
                "02/03/2026",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="date_format is required",
    ):
        prepare_temporal_axis(
            data,
            "Date",
        )


def test_prepare_temporal_axis_combines_date_and_time() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "10/03/2004",
                "10/03/2004",
            ],
            "Time": [
                "18.00.00",
                "19.30.00",
            ],
        }
    )

    result = prepare_temporal_axis(
        data,
        "Date",
        date_format="%d/%m/%Y",
        time_column="Time",
        time_format="%H.%M.%S",
    )

    assert result.timestamps.iloc[0] == pd.Timestamp("2004-03-10 18:00:00")

    assert result.timestamps.iloc[1] == pd.Timestamp("2004-03-10 19:30:00")


def test_prepare_temporal_axis_requires_format_for_text_time() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
            ],
            "Time": [
                "10:30:00",
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="time_format is required",
    ):
        prepare_temporal_axis(
            data,
            "Date",
            date_format="%Y-%m-%d",
            time_column="Time",
        )


def test_prepare_temporal_axis_counts_invalid_rows() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
                "invalid",
                None,
            ]
        }
    )

    result = prepare_temporal_axis(
        data,
        "Date",
        date_format="%Y-%m-%d",
    )

    assert result.total_count == 3
    assert result.valid_count == 1
    assert result.invalid_count == 2


def test_prepare_temporal_axis_requires_distinct_date_and_time_columns() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="must be different",
    ):
        prepare_temporal_axis(
            data,
            "Date",
            date_format="%Y-%m-%d",
            time_column="Date",
            time_format="%H:%M:%S",
        )


def test_prepare_temporal_axis_does_not_modify_source_data() -> None:
    data = pd.DataFrame(
        {
            "Date": [
                "2026-01-01",
                "2026-01-02",
            ],
            "value": [
                1,
                2,
            ],
        }
    )

    original = data.copy(deep=True)

    prepare_temporal_axis(
        data,
        "Date",
        date_format="%Y-%m-%d",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )


def test_sampling_summary_reports_duplicates_and_order() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 02:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 01:00:00",
                ]
            )
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    result = analyze_temporal_sampling(temporal)

    assert result.valid_count == 3
    assert result.unique_timestamp_count == 2
    assert result.duplicate_row_count == 1
    assert result.duplicate_timestamp_count == 1
    assert result.is_chronologically_sorted is False


def test_sampling_summary_reports_regular_interval() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-01 03:00:00",
                ]
            )
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    result = analyze_temporal_sampling(temporal)

    expected = pd.Timedelta(hours=1)

    assert result.interval_count == 3
    assert result.minimum_interval == expected
    assert result.median_interval == expected
    assert result.maximum_interval == expected
    assert result.dominant_interval == expected
    assert result.dominant_interval_count == 3
    assert result.dominant_interval_ratio == pytest.approx(1.0)


def test_detect_time_gaps_uses_explicit_expected_interval() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2026-01-01 04:00:00",
                ]
            )
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    result = detect_time_gaps(
        temporal,
        "1h",
    )

    assert result.gap_count == 1

    gap = result.gaps[0]

    assert gap.start == pd.Timestamp("2026-01-01 01:00:00")

    assert gap.end == pd.Timestamp("2026-01-01 04:00:00")

    assert gap.duration == pd.Timedelta(hours=3)

    assert gap.excess_duration == pd.Timedelta(hours=2)


def test_detect_time_gaps_rejects_invalid_expected_interval() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            )
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        detect_time_gaps(
            temporal,
            "0s",
        )


def test_prepare_temporal_numeric_series_sorts_and_excludes_invalid_values() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-03",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-04",
                ]
            ),
            "value": [
                3.0,
                1.0,
                np.inf,
                np.nan,
            ],
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    result = prepare_temporal_numeric_series(
        data,
        temporal,
        "value",
    )

    assert result.total_count == 4
    assert result.valid_count == 2
    assert result.excluded_count == 2

    assert list(result.data["timestamp"]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-01-03"),
    ]

    assert list(result.data["value"]) == [
        1.0,
        3.0,
    ]


def test_daily_mean_aggregation() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01 01:00:00",
                    "2026-01-01 02:00:00",
                    "2026-01-02 01:00:00",
                ]
            ),
            "value": [
                1.0,
                3.0,
                5.0,
            ],
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    result = aggregate_temporal_numeric_series(
        data,
        temporal,
        "value",
        frequency="day",
        aggregation="mean",
    )

    assert result.frequency == TemporalFrequency.DAY
    assert result.aggregation == TemporalAggregation.MEAN
    assert result.bucket_count == 2

    assert list(result.data["value"]) == pytest.approx(
        [
            2.0,
            5.0,
        ]
    )

    assert list(result.data["count"]) == [
        2,
        1,
    ]


def test_monthly_sum_aggregation() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-31",
                    "2026-02-01",
                ]
            ),
            "value": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    result = aggregate_temporal_numeric_series(
        data,
        temporal,
        "value",
        frequency=TemporalFrequency.MONTH,
        aggregation=TemporalAggregation.SUM,
    )

    assert result.bucket_count == 2

    assert list(result.data["timestamp"]) == [
        pd.Timestamp("2026-01-01"),
        pd.Timestamp("2026-02-01"),
    ]

    assert list(result.data["value"]) == pytest.approx(
        [
            3.0,
            3.0,
        ]
    )


def test_temporal_aggregation_rejects_non_numeric_value_column() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "city": [
                "Delhi",
                "Mumbai",
            ],
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    with pytest.raises(
        ValueError,
        match="must be numeric",
    ):
        aggregate_temporal_numeric_series(
            data,
            temporal,
            "city",
        )


def test_temporal_aggregation_rejects_unsupported_configuration() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "value": [
                1.0,
                2.0,
            ],
        }
    )

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported temporal frequency",
    ):
        aggregate_temporal_numeric_series(
            data,
            temporal,
            "value",
            frequency="quarter",
        )

    with pytest.raises(
        ValueError,
        match="Unsupported temporal aggregation",
    ):
        aggregate_temporal_numeric_series(
            data,
            temporal,
            "value",
            aggregation="variance",
        )


def test_temporal_aggregation_does_not_modify_source_data() -> None:
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-02",
                    "2026-01-01",
                ]
            ),
            "value": [
                2.0,
                1.0,
            ],
        }
    )

    original = data.copy(deep=True)

    temporal = prepare_temporal_axis(
        data,
        "timestamp",
    )

    aggregate_temporal_numeric_series(
        data,
        temporal,
        "value",
    )

    pd.testing.assert_frame_equal(
        data,
        original,
    )
