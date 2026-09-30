import pandas as pd
import streamlit as st

from data_diagnostics.analysis import profile_dataset
from data_diagnostics.ingestion import SemanticType


def display_value(value: object) -> str:
    """Format a summary value for stable display."""

    if value is None:
        return "—"

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


def build_schema_table(profile) -> pd.DataFrame:
    """Convert schema metadata into a display table."""

    rows = []

    for column in profile.columns:
        schema = column.schema

        rows.append(
            {
                "Column": schema.name,
                "Physical dtype": schema.physical_dtype,
                "Semantic type": schema.semantic_type.value,
                "Missing": schema.missing_count,
                "Missing %": f"{schema.missing_ratio:.1%}",
                "Unique": schema.unique_count,
                "Unique %": f"{schema.unique_ratio:.1%}",
                "Confidence": f"{schema.confidence:.0%}",
                "Temporal format": (schema.temporal_format or "—"),
                "Reason": schema.reason,
            }
        )

    return pd.DataFrame(rows)


def build_quality_table(profile) -> pd.DataFrame:
    """Convert quality findings into a display table."""

    rows = []

    for issue in profile.quality.issues:
        rows.append(
            {
                "Severity": issue.severity.value,
                "Scope": issue.column or "Dataset",
                "Finding": issue.code.value,
                "Observed value": display_value(issue.observed_value),
                "Affected": display_value(issue.affected_count),
                "Affected %": (
                    f"{issue.affected_ratio:.1%}" if issue.affected_ratio is not None else "—"
                ),
                "Message": issue.message,
            }
        )

    return pd.DataFrame(rows)


def render_statistic_table(
    rows: list[tuple[str, object]],
) -> None:
    """Render descriptive statistics."""

    st.dataframe(
        pd.DataFrame(
            {
                "Statistic": [name for name, _ in rows],
                "Value": [display_value(value) for _, value in rows],
            }
        ),
        hide_index=True,
        width="stretch",
    )


def render_column_summary(
    column,
    quality_issues,
) -> None:
    """Render one column profile."""

    schema = column.schema

    st.markdown(
        f"**Semantic type:** `{schema.semantic_type.value}`  \n"
        f"**Physical dtype:** `{schema.physical_dtype}`  \n"
        f"**Confidence:** `{schema.confidence:.0%}`"
    )

    if schema.temporal_format is not None:
        st.markdown(f"**Detected temporal format:** `{schema.temporal_format}`")

    st.caption(schema.reason)

    sentinel_issues = [issue for issue in quality_issues if issue.code.value == "possible_sentinel"]

    if sentinel_issues:
        values = ", ".join(display_value(issue.observed_value) for issue in sentinel_issues)

        st.warning(
            "Possible sentinel value(s): "
            f"{values}. Statistics shown here are based "
            "on the currently selected dataset."
        )

    if column.numeric is not None:
        summary = column.numeric

        render_statistic_table(
            [
                ("Count", summary.count),
                ("Mean", summary.mean),
                ("Median", summary.median),
                (
                    "Standard deviation",
                    summary.standard_deviation,
                ),
                ("Minimum", summary.minimum),
                ("Maximum", summary.maximum),
            ]
        )

    elif column.categorical is not None:
        summary = column.categorical

        if schema.semantic_type == SemanticType.IDENTIFIER:
            rows = [
                ("Count", summary.count),
                (
                    "Unique values",
                    summary.unique_count,
                ),
            ]

        else:
            rows = [
                ("Count", summary.count),
                (
                    "Unique values",
                    summary.unique_count,
                ),
                (
                    "Most frequent value",
                    summary.most_frequent_value,
                ),
                (
                    "Most frequent count",
                    summary.most_frequent_count,
                ),
            ]

        render_statistic_table(rows)

    elif column.datetime is not None:
        summary = column.datetime

        if summary.earliest is None or summary.latest is None:
            st.warning(
                "The exact date order is ambiguous. "
                "Confirm the format before interpreting its range."
            )

        render_statistic_table(
            [
                ("Count", summary.count),
                ("Earliest", summary.earliest),
                ("Latest", summary.latest),
            ]
        )

    elif column.time is not None:
        summary = column.time

        render_statistic_table(
            [
                ("Count", summary.count),
                ("Earliest", summary.earliest),
                ("Latest", summary.latest),
            ]
        )

    else:
        st.info("No descriptive summary is available for this column.")


st.title("Dataset Profile")


if st.session_state.raw_data is None:
    st.warning("No dataset is loaded. Open Workspace and upload a dataset first.")
    st.stop()


dataset_view = st.radio(
    "Profile",
    options=["Working dataset", "Raw dataset"],
    horizontal=True,
)


if dataset_view == "Raw dataset":
    data = st.session_state.raw_data

    profile = st.session_state.raw_profile

else:
    data = st.session_state.working_data

    profile = st.session_state.working_profile


if profile is None:
    profile = profile_dataset(data)


st.caption(
    "Raw data remains unchanged. The working dataset reflects "
    "only transformations explicitly applied in the Transform page."
)


st.subheader("Overview")

metric_columns = st.columns(5)

metric_columns[0].metric(
    "Rows",
    f"{profile.row_count:,}",
)

metric_columns[1].metric(
    "Columns",
    f"{profile.column_count:,}",
)

metric_columns[2].metric(
    "Missing Cells",
    f"{profile.quality.missing_cell_count:,}",
)

metric_columns[3].metric(
    "Empty Rows",
    f"{profile.quality.all_missing_row_count:,}",
)

metric_columns[4].metric(
    "Quality Findings",
    profile.quality.issue_count,
)


st.subheader("Data Preview")

st.dataframe(
    data.head(20),
    width="stretch",
)


st.subheader("Schema")

st.dataframe(
    build_schema_table(profile),
    hide_index=True,
    width="stretch",
)


st.subheader("Data Quality")

if profile.quality.issue_count == 0:
    st.success("No quality findings were detected by the current checks.")

else:
    st.dataframe(
        build_quality_table(profile),
        hide_index=True,
        width="stretch",
    )


st.subheader("Column Profiles")

for column in profile.columns:
    column_quality_issues = [
        issue for issue in profile.quality.issues if issue.column == column.schema.name
    ]

    with st.expander(column.schema.name):
        render_column_summary(
            column,
            column_quality_issues,
        )
