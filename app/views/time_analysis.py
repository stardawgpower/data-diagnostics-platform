import numpy as np
import pandas as pd
import streamlit as st

from data_diagnostics.analysis import (
    TemporalAggregation,
    TemporalFrequency,
    aggregate_temporal_numeric_series,
    analyze_temporal_sampling,
    detect_time_gaps,
    prepare_temporal_axis,
    prepare_temporal_numeric_series,
)
from data_diagnostics.ingestion import SemanticType


def columns_by_semantic_type(
    profile,
    semantic_type: SemanticType,
) -> list[str]:
    """Return columns matching one semantic type."""

    return [
        column.schema.name
        for column in profile.columns
        if column.schema.semantic_type == semantic_type
    ]


def numeric_columns_from_profile(profile) -> list[str]:
    """Return numeric columns available for temporal analysis."""

    numeric_types = {
        SemanticType.NUMERIC_DISCRETE,
        SemanticType.NUMERIC_CONTINUOUS,
    }

    return [
        column.schema.name
        for column in profile.columns
        if column.schema.semantic_type in numeric_types
    ]


def schema_by_name(profile) -> dict[str, object]:
    """Return column schemas indexed by column name."""

    return {column.schema.name: column.schema for column in profile.columns}


def is_native_datetime(
    data: pd.DataFrame,
    column: str,
) -> bool:
    """Return whether a column has a native pandas datetime dtype."""

    return pd.api.types.is_datetime64_any_dtype(data[column].dtype)


def format_timestamp(
    value: pd.Timestamp | None,
) -> str:
    """Format an optional timestamp for display."""

    if value is None:
        return "—"

    return value.isoformat(
        sep=" ",
    )


def format_timedelta(
    value: pd.Timedelta | None,
) -> str:
    """Format an optional duration for display."""

    if value is None:
        return "—"

    return str(value)


def downsample_chronological(
    data: pd.DataFrame,
    *,
    max_points: int = 5_000,
) -> pd.DataFrame:
    """Deterministically reduce chart rows while preserving time coverage."""

    if len(data) <= max_points:
        return data

    positions = np.linspace(
        0,
        len(data) - 1,
        num=max_points,
        dtype=int,
    )

    return data.iloc[np.unique(positions)].copy()


def aggregation_label(
    aggregation: TemporalAggregation,
) -> str:
    """Return a human-readable aggregation label."""

    labels = {
        TemporalAggregation.MEAN: "Mean",
        TemporalAggregation.MEDIAN: "Median",
        TemporalAggregation.SUM: "Sum",
        TemporalAggregation.MINIMUM: "Minimum",
        TemporalAggregation.MAXIMUM: "Maximum",
    }

    return labels[aggregation]


def frequency_label(
    frequency: TemporalFrequency,
) -> str:
    """Return a human-readable temporal frequency label."""

    labels = {
        TemporalFrequency.HOUR: "Hour",
        TemporalFrequency.DAY: "Day",
        TemporalFrequency.WEEK: "Week",
        TemporalFrequency.MONTH: "Month",
    }

    return labels[frequency]


st.title("Time Analysis")

st.write(
    "Inspect temporal coverage, sampling intervals, gaps, "
    "chronological numeric series, and calendar aggregations."
)

st.caption(
    "Time analysis uses the current working dataset. "
    "Analysis may sort derived views chronologically, but the working "
    "dataset itself is never reordered or modified."
)


working_data = st.session_state.working_data
working_profile = st.session_state.working_profile

dataset_fingerprint = st.session_state.get("dataset_fingerprint") or "dataset"

widget_prefix = f"time_analysis::{dataset_fingerprint}"


if working_data is None:
    st.warning("No dataset is loaded. Open Workspace and upload a dataset first.")
    st.stop()


if working_profile is None:
    st.warning(
        "The working dataset profile is unavailable. Return to Workspace and reload the dataset."
    )
    st.stop()


datetime_columns = columns_by_semantic_type(
    working_profile,
    SemanticType.DATETIME,
)

time_columns = columns_by_semantic_type(
    working_profile,
    SemanticType.TIME,
)

numeric_columns = numeric_columns_from_profile(
    working_profile,
)

schemas = schema_by_name(
    working_profile,
)


if not datetime_columns:
    st.info("No datetime-like column was detected in the working dataset.")
    st.stop()


st.subheader("Temporal Axis")

axis_columns = st.columns(
    [
        2,
        2,
    ]
)


with axis_columns[0]:
    date_column = st.selectbox(
        "Date or datetime column",
        options=datetime_columns,
        key=f"{widget_prefix}::date_column",
    )


date_schema = schemas[date_column]

native_date = is_native_datetime(
    working_data,
    date_column,
)

inferred_date_format = (
    date_schema.temporal_format if date_schema.temporal_format is not None else ""
)


with axis_columns[1]:
    if time_columns:
        use_time_column = st.checkbox(
            "Combine with a separate time column",
            value=False,
            help=("Use this when the dataset stores date and time-of-day in separate columns."),
            key=f"{widget_prefix}::use_time_column",
        )

    else:
        use_time_column = False

        st.checkbox(
            "Combine with a separate time column",
            value=False,
            disabled=True,
            help=("No separate time-of-day column was detected."),
            key=(f"{widget_prefix}::use_time_column_unavailable"),
        )


format_columns = st.columns(2)


with format_columns[0]:
    if native_date:
        st.text_input(
            "Date format",
            value="Native datetime dtype",
            disabled=True,
            key=(f"{widget_prefix}::native_date_format::{date_column}"),
        )

        date_format = None

    else:
        date_format_input = st.text_input(
            "Date format",
            value=inferred_date_format,
            placeholder="%d/%m/%Y",
            help=(
                "Textual date columns require an explicit format. "
                "Inferred unambiguous formats are prefilled. "
                "Ambiguous dates are never guessed."
            ),
            key=(f"{widget_prefix}::date_format::{date_column}"),
        )

        date_format = date_format_input.strip() or None


time_column = None
time_format = None


with format_columns[1]:
    if use_time_column:
        time_column = st.selectbox(
            "Time column",
            options=time_columns,
            key=(f"{widget_prefix}::time_column::{date_column}"),
        )

        time_schema = schemas[time_column]

        inferred_time_format = (
            time_schema.temporal_format if time_schema.temporal_format is not None else ""
        )

        time_format_input = st.text_input(
            "Time format",
            value=inferred_time_format,
            placeholder="%H:%M:%S",
            help=(
                "Textual time columns require an explicit format. "
                "Inferred formats are prefilled when available."
            ),
            key=(f"{widget_prefix}::time_format::{time_column}"),
        )

        time_format = time_format_input.strip() or None

    else:
        st.caption("Using the selected date/datetime column as the complete temporal axis.")


try:
    temporal = prepare_temporal_axis(
        working_data,
        date_column,
        date_format=date_format,
        time_column=time_column,
        time_format=time_format,
    )

except (
    TypeError,
    ValueError,
) as exc:
    st.error(str(exc))

    st.stop()


sampling = analyze_temporal_sampling(temporal)


overview_tab, gaps_tab, series_tab = st.tabs(
    [
        "Sampling",
        "Gaps",
        "Numeric Series",
    ]
)


with overview_tab:
    st.subheader("Temporal Coverage")

    coverage_metrics = st.columns(4)

    coverage_metrics[0].metric(
        "Valid Timestamps",
        f"{sampling.valid_count:,}",
    )

    coverage_metrics[1].metric(
        "Invalid / Missing",
        f"{sampling.invalid_count:,}",
    )

    coverage_metrics[2].metric(
        "Unique Timestamps",
        f"{sampling.unique_timestamp_count:,}",
    )

    coverage_metrics[3].metric(
        "Duplicate Timestamp Rows",
        f"{sampling.duplicate_row_count:,}",
    )

    coverage_frame = pd.DataFrame(
        [
            {
                "Diagnostic": "Earliest timestamp",
                "Value": format_timestamp(sampling.earliest),
            },
            {
                "Diagnostic": "Latest timestamp",
                "Value": format_timestamp(sampling.latest),
            },
            {
                "Diagnostic": "Coverage span",
                "Value": format_timedelta(sampling.span),
            },
            {
                "Diagnostic": "Chronologically sorted",
                "Value": ("Yes" if sampling.is_chronologically_sorted else "No"),
            },
            {
                "Diagnostic": "Duplicate timestamp values",
                "Value": (f"{sampling.duplicate_timestamp_count:,}"),
            },
        ]
    )

    st.dataframe(
        coverage_frame,
        hide_index=True,
        width="stretch",
    )

    if not sampling.is_chronologically_sorted:
        st.info(
            "The source rows are not in chronological order. "
            "Temporal analysis sorts derived series only; "
            "the working dataset remains unchanged."
        )

    if sampling.duplicate_row_count > 0:
        st.info(
            "Duplicate timestamps are present. "
            "They remain valid observations and are not automatically removed."
        )

    st.divider()

    st.subheader("Sampling Intervals")

    if sampling.interval_count == 0:
        st.info("At least two unique valid timestamps are required for interval diagnostics.")

    else:
        interval_metrics = st.columns(4)

        interval_metrics[0].metric(
            "Intervals",
            f"{sampling.interval_count:,}",
        )

        interval_metrics[1].metric(
            "Minimum",
            format_timedelta(sampling.minimum_interval),
        )

        interval_metrics[2].metric(
            "Median",
            format_timedelta(sampling.median_interval),
        )

        interval_metrics[3].metric(
            "Maximum",
            format_timedelta(sampling.maximum_interval),
        )

        if sampling.dominant_interval is not None:
            st.markdown("#### Dominant Interval")

            dominant_metrics = st.columns(3)

            dominant_metrics[0].metric(
                "Interval",
                format_timedelta(sampling.dominant_interval),
            )

            dominant_metrics[1].metric(
                "Occurrences",
                f"{sampling.dominant_interval_count:,}",
            )

            dominant_metrics[2].metric(
                "Share of intervals",
                (
                    f"{sampling.dominant_interval_ratio:.2%}"
                    if sampling.dominant_interval_ratio is not None
                    else "—"
                ),
            )

        else:
            st.info("No unique dominant sampling interval was identified.")


with gaps_tab:
    st.subheader("Gap Detection")

    st.caption(
        "A gap is reported only when the interval between consecutive "
        "unique timestamps is larger than the expected interval you specify."
    )

    suggested_interval = (
        str(sampling.dominant_interval) if sampling.dominant_interval is not None else ""
    )

    expected_interval_text = st.text_input(
        "Expected interval",
        value=suggested_interval,
        placeholder="1h",
        help=(
            "Examples: 30min, 1h, 1D. "
            "When a unique dominant sampling interval exists, "
            "it is suggested automatically but remains editable."
        ),
        key=(f"{widget_prefix}::expected_interval::{date_column}::{time_column or 'none'}"),
    )

    if not expected_interval_text.strip():
        st.info("Enter an expected interval to run gap detection.")

    else:
        try:
            gaps = detect_time_gaps(
                temporal,
                expected_interval_text.strip(),
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            st.error(str(exc))

        else:
            gap_metrics = st.columns(3)

            gap_metrics[0].metric(
                "Expected Interval",
                str(gaps.expected_interval),
            )

            gap_metrics[1].metric(
                "Unique Timestamps",
                f"{gaps.unique_timestamp_count:,}",
            )

            gap_metrics[2].metric(
                "Detected Gaps",
                f"{gaps.gap_count:,}",
            )

            if not gaps.gaps:
                st.success("No intervals exceed the selected expected interval.")

            else:
                gaps_frame = pd.DataFrame(
                    [
                        {
                            "Start": gap.start,
                            "End": gap.end,
                            "Duration": str(gap.duration),
                            "Excess duration": str(gap.excess_duration),
                        }
                        for gap in gaps.gaps
                    ]
                )

                st.dataframe(
                    gaps_frame,
                    hide_index=True,
                    width="stretch",
                )


with series_tab:
    st.subheader("Numeric Time Series")

    if not numeric_columns:
        st.info("No numeric columns are available for temporal analysis.")

    else:
        value_column = st.selectbox(
            "Numeric column",
            options=numeric_columns,
            key=f"{widget_prefix}::value_column",
        )

        temporal_series = prepare_temporal_numeric_series(
            working_data,
            temporal,
            value_column,
        )

        series_metrics = st.columns(3)

        series_metrics[0].metric(
            "Total Rows",
            f"{temporal_series.total_count:,}",
        )

        series_metrics[1].metric(
            "Valid Observations",
            f"{temporal_series.valid_count:,}",
        )

        series_metrics[2].metric(
            "Excluded Rows",
            f"{temporal_series.excluded_count:,}",
        )

        if temporal_series.data.empty:
            st.info(
                "No finite numeric observations with valid timestamps "
                "are available for this column."
            )

        else:
            st.markdown("#### Chronological Series")

            chart_data = downsample_chronological(temporal_series.data)

            if len(chart_data) < len(temporal_series.data):
                st.caption(
                    "The chart uses a deterministic time-spanning sample "
                    "of up to 5,000 observations for responsiveness. "
                    "Counts above use the full valid series."
                )

            raw_chart = chart_data.set_index("timestamp")[["value"]]

            st.line_chart(
                raw_chart,
                width="stretch",
            )

            with st.expander("View chronological observations"):
                st.dataframe(
                    temporal_series.data.head(1_000),
                    hide_index=True,
                    width="stretch",
                )

                if len(temporal_series.data) > 1_000:
                    st.caption("Preview limited to the first 1,000 chronological observations.")

            st.divider()

            st.markdown("#### Calendar Aggregation")

            aggregation_columns = st.columns(2)

            with aggregation_columns[0]:
                frequency_label_value = st.selectbox(
                    "Frequency",
                    options=[
                        "Hour",
                        "Day",
                        "Week",
                        "Month",
                    ],
                    index=1,
                    key=f"{widget_prefix}::frequency",
                )

            frequency_lookup = {
                "Hour": TemporalFrequency.HOUR,
                "Day": TemporalFrequency.DAY,
                "Week": TemporalFrequency.WEEK,
                "Month": TemporalFrequency.MONTH,
            }

            frequency = frequency_lookup[frequency_label_value]

            with aggregation_columns[1]:
                aggregation_label_value = st.selectbox(
                    "Aggregation",
                    options=[
                        "Mean",
                        "Median",
                        "Sum",
                        "Minimum",
                        "Maximum",
                    ],
                    key=f"{widget_prefix}::aggregation",
                )

            aggregation_lookup = {
                "Mean": TemporalAggregation.MEAN,
                "Median": TemporalAggregation.MEDIAN,
                "Sum": TemporalAggregation.SUM,
                "Minimum": TemporalAggregation.MINIMUM,
                "Maximum": TemporalAggregation.MAXIMUM,
            }

            aggregation = aggregation_lookup[aggregation_label_value]

            aggregated = aggregate_temporal_numeric_series(
                working_data,
                temporal,
                value_column,
                frequency=frequency,
                aggregation=aggregation,
            )

            aggregate_metrics = st.columns(3)

            aggregate_metrics[0].metric(
                "Frequency",
                frequency_label(aggregated.frequency),
            )

            aggregate_metrics[1].metric(
                "Aggregation",
                aggregation_label(aggregated.aggregation),
            )

            aggregate_metrics[2].metric(
                "Buckets",
                f"{aggregated.bucket_count:,}",
            )

            if aggregated.data.empty:
                st.info("No aggregated observations are available.")

            else:
                aggregate_chart = aggregated.data.set_index("timestamp")[["value"]]

                st.line_chart(
                    aggregate_chart,
                    width="stretch",
                )

                with st.expander("View aggregated observations"):
                    st.dataframe(
                        aggregated.data,
                        hide_index=True,
                        width="stretch",
                    )
