import pandas as pd
import streamlit as st

from data_diagnostics.analysis import profile_dataset
from data_diagnostics.ingestion import DatasetError, SemanticType, read_dataset

st.set_page_config(
    page_title="Data Diagnostics Platform",
    page_icon="📊",
    layout="wide",
)


def _display_value(value: object) -> str:
    """Format a summary value for stable Streamlit and PDF rendering."""

    if value is None:
        return "—"

    if isinstance(value, float):
        return f"{value:.6g}"

    return str(value)


def build_schema_table(profile) -> pd.DataFrame:
    """Convert inferred schema metadata into a display table."""

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
                "Temporal format": schema.temporal_format or "—",
                "Reason": schema.reason,
            }
        )

    return pd.DataFrame(rows)


def build_quality_table(profile) -> pd.DataFrame:
    """Convert quality issues into a display table."""

    rows = [
        {
            "Severity": issue.severity.value,
            "Scope": issue.column or "Dataset",
            "Issue": issue.code.value,
            "Observed value": _display_value(issue.observed_value),
            "Affected": _display_value(issue.affected_count),
            "Affected %": (
                f"{issue.affected_ratio:.1%}" if issue.affected_ratio is not None else "—"
            ),
            "Message": issue.message,
        }
        for issue in profile.quality.issues
    ]

    return pd.DataFrame(rows)


def _render_statistic_table(rows: list[tuple[str, object]]) -> None:
    """Render a two-column statistic table using display-safe strings."""

    st.dataframe(
        pd.DataFrame(
            {
                "Statistic": [name for name, _ in rows],
                "Value": [_display_value(value) for _, value in rows],
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


def render_column_summary(column, quality_issues) -> None:
    """Render descriptive information for one profiled column."""

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
        values = ", ".join(_display_value(issue.observed_value) for issue in sentinel_issues)
        st.warning(
            "Possible sentinel value(s) were detected in this column: "
            f"{values}. The statistics below are calculated from the raw "
            "uploaded values; nothing has been replaced automatically."
        )

    if column.numeric is not None:
        summary = column.numeric
        _render_statistic_table(
            [
                ("Count", summary.count),
                ("Mean", summary.mean),
                ("Median", summary.median),
                ("Standard deviation", summary.standard_deviation),
                ("Minimum", summary.minimum),
                ("Maximum", summary.maximum),
            ]
        )

    elif column.categorical is not None:
        summary = column.categorical

        if schema.semantic_type == SemanticType.IDENTIFIER:
            rows = [
                ("Count", summary.count),
                ("Unique values", summary.unique_count),
            ]
        else:
            rows = [
                ("Count", summary.count),
                ("Unique values", summary.unique_count),
                ("Most frequent value", summary.most_frequent_value),
                ("Most frequent count", summary.most_frequent_count),
            ]

        _render_statistic_table(rows)

    elif column.datetime is not None:
        summary = column.datetime

        if summary.earliest is None or summary.latest is None:
            st.warning(
                "The column appears to contain dates, but the exact date format "
                "is ambiguous. Confirm the format before interpreting its range."
            )

        _render_statistic_table(
            [
                ("Count", summary.count),
                ("Earliest", summary.earliest),
                ("Latest", summary.latest),
            ]
        )

    elif column.time is not None:
        summary = column.time

        if summary.earliest is None or summary.latest is None:
            st.warning(
                "The column appears to contain times, but the exact time format "
                "is ambiguous. Confirm the format before interpreting its range."
            )

        _render_statistic_table(
            [
                ("Count", summary.count),
                ("Earliest", summary.earliest),
                ("Latest", summary.latest),
            ]
        )

    else:
        st.info("No descriptive summary is available for this column.")


st.title("Data Diagnostics Platform")

st.write("Upload a dataset to inspect its schema, data quality, and descriptive profile.")

st.caption(
    "The uploaded dataset is profiled without silently deleting rows, columns, "
    "or replacing suspicious values."
)

uploaded_file = st.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx"],
    help="Currently supported formats: CSV and XLSX.",
)

if uploaded_file is None:
    st.info("Upload a CSV or XLSX file to begin.")
    st.stop()


try:
    uploaded_file.seek(0)

    data = read_dataset(
        uploaded_file,
        filename=uploaded_file.name,
    )
except DatasetError as exc:
    st.error(str(exc))
    st.stop()


profile = profile_dataset(data)


st.subheader("Dataset Overview")

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
    "Quality Issues",
    profile.quality.issue_count,
)


st.subheader("Data Preview")

st.dataframe(
    data.head(20),
    use_container_width=True,
)


st.subheader("Schema")

schema_table = build_schema_table(profile)

st.dataframe(
    schema_table,
    hide_index=True,
    use_container_width=True,
)


st.subheader("Data Quality")

if profile.quality.issue_count == 0:
    st.success("No quality issues were detected by the current checks.")
else:
    quality_table = build_quality_table(profile)

    st.dataframe(
        quality_table,
        hide_index=True,
        use_container_width=True,
    )

    sentinel_count = sum(
        issue.code.value == "possible_sentinel" for issue in profile.quality.issues
    )

    if sentinel_count:
        st.info(
            f"{sentinel_count} possible sentinel-value pattern(s) were detected. "
            "These values are flagged for review and have not been replaced."
        )


st.subheader("Column Profiles")

for column in profile.columns:
    column_quality_issues = [
        issue for issue in profile.quality.issues if issue.column == column.schema.name
    ]

    with st.expander(column.schema.name):
        render_column_summary(column, column_quality_issues)
