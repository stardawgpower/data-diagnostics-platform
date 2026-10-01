import pandas as pd
import streamlit as st

from data_diagnostics.analysis import (
    PivotAggregation,
    build_pivot_table,
    estimate_pivot_cardinality,
)
from data_diagnostics.ingestion import SemanticType


def numeric_columns_from_profile(profile) -> list[str]:
    """Return numeric columns suitable for numeric aggregation."""

    numeric_types = {
        SemanticType.NUMERIC_DISCRETE,
        SemanticType.NUMERIC_CONTINUOUS,
    }

    return [
        column.schema.name
        for column in profile.columns
        if column.schema.semantic_type in numeric_types
    ]


def semantic_types_by_name(profile) -> dict[str, SemanticType]:
    """Return semantic types indexed by column name."""

    return {column.schema.name: column.schema.semantic_type for column in profile.columns}


def column_label(
    column: str,
    semantic_types: dict[str, SemanticType],
) -> str:
    """Return a display label with semantic-type context."""

    semantic_type = semantic_types.get(column)

    if semantic_type is None:
        return column

    return f"{column} ({semantic_type.value.replace('_', ' ')})"


def aggregation_label(
    aggregation: PivotAggregation,
) -> str:
    """Return a human-readable aggregation name."""

    labels = {
        PivotAggregation.ROW_COUNT: "Row count",
        PivotAggregation.COUNT: "Non-missing count",
        PivotAggregation.SUM: "Sum",
        PivotAggregation.MEAN: "Mean",
        PivotAggregation.MEDIAN: "Median",
        PivotAggregation.MINIMUM: "Minimum",
        PivotAggregation.MAXIMUM: "Maximum",
    }

    return labels[aggregation]


st.title("Pivot Analysis")

st.write(
    "Summarize the working dataset across one or more dimensions "
    "using generic grouped and pivoted views."
)

st.caption(
    "Pivot analysis is descriptive only. "
    "It never modifies the working dataset, fills missing combinations, "
    "or silently reduces dimension cardinality."
)


working_data = st.session_state.working_data
working_profile = st.session_state.working_profile

dataset_fingerprint = st.session_state.get("dataset_fingerprint") or "dataset"

widget_prefix = f"pivot_analysis::{dataset_fingerprint}"


if working_data is None:
    st.warning("No dataset is loaded. Open Workspace and upload a dataset first.")
    st.stop()


if working_profile is None:
    st.warning(
        "The working dataset profile is unavailable. Return to Workspace and reload the dataset."
    )
    st.stop()


all_columns = [str(column) for column in working_data.columns]

numeric_columns = numeric_columns_from_profile(working_profile)

semantic_types = semantic_types_by_name(working_profile)


if not all_columns:
    st.info("The working dataset has no columns available for pivot analysis.")
    st.stop()


st.subheader("Pivot Configuration")

row_dimensions = st.multiselect(
    "Row dimensions",
    options=all_columns,
    default=[],
    format_func=lambda column: column_label(
        column,
        semantic_types,
    ),
    help=("Select one or more columns that define the rows of the grouped result."),
    key=f"{widget_prefix}::row_dimensions",
)


available_column_dimensions = [column for column in all_columns if column not in row_dimensions]

column_dimensions = st.multiselect(
    "Column dimensions",
    options=available_column_dimensions,
    default=[],
    format_func=lambda column: column_label(
        column,
        semantic_types,
    ),
    help=(
        "Optional columns that become pivoted matrix columns. "
        "High-cardinality combinations may create very wide tables."
    ),
    key=f"{widget_prefix}::column_dimensions",
)


dimension_columns = row_dimensions + column_dimensions


aggregation_options = [
    PivotAggregation.ROW_COUNT,
    PivotAggregation.COUNT,
    PivotAggregation.SUM,
    PivotAggregation.MEAN,
    PivotAggregation.MEDIAN,
    PivotAggregation.MINIMUM,
    PivotAggregation.MAXIMUM,
]

aggregation = st.selectbox(
    "Aggregation",
    options=aggregation_options,
    format_func=aggregation_label,
    help=(
        "Row count counts source rows per group. "
        "Non-missing count counts non-null values in a selected column. "
        "Other aggregations require a numeric value column."
    ),
    key=f"{widget_prefix}::aggregation",
)


value_column = None

if aggregation == PivotAggregation.ROW_COUNT:
    st.caption("Row count does not require a value column.")

elif aggregation == PivotAggregation.COUNT:
    count_columns = [column for column in all_columns if column not in dimension_columns]

    if not count_columns:
        st.warning(
            "No remaining column is available to count after excluding the selected dimensions."
        )
        st.stop()

    value_column = st.selectbox(
        "Value column",
        options=count_columns,
        format_func=lambda column: column_label(
            column,
            semantic_types,
        ),
        help=("Counts non-missing values from this column within each dimension group."),
        key=f"{widget_prefix}::count_value_column",
    )

else:
    numeric_value_columns = [
        column for column in numeric_columns if column not in dimension_columns
    ]

    if not numeric_value_columns:
        st.warning(
            "No numeric column is available for the selected "
            "aggregation after excluding the pivot dimensions."
        )
        st.stop()

    value_column = st.selectbox(
        "Numeric value column",
        options=numeric_value_columns,
        format_func=lambda column: column_label(
            column,
            semantic_types,
        ),
        help=("Numeric measure aggregated within each dimension group."),
        key=f"{widget_prefix}::numeric_value_column",
    )


option_columns = st.columns(2)


with option_columns[0]:
    include_missing_dimensions = st.checkbox(
        "Include missing dimension values",
        value=False,
        help=(
            "When disabled, rows with missing values in any selected "
            "dimension are excluded from the pivot analysis."
        ),
        key=f"{widget_prefix}::include_missing_dimensions",
    )


with option_columns[1]:
    max_cells = st.number_input(
        "Maximum potential matrix cells",
        min_value=100,
        max_value=100_000,
        value=10_000,
        step=100,
        help=(
            "Safety limit for the potential dense pivot matrix. "
            "The analysis stops before building a matrix larger "
            "than this threshold."
        ),
        key=f"{widget_prefix}::max_cells",
    )


if not row_dimensions:
    st.info("Select at least one row dimension to run pivot analysis.")
    st.stop()


try:
    cardinality = estimate_pivot_cardinality(
        working_data,
        row_dimensions,
        column_dimensions,
        include_missing_dimensions=(include_missing_dimensions),
        max_cells=int(max_cells),
    )

except (
    TypeError,
    ValueError,
) as exc:
    st.error(str(exc))
    st.stop()


st.subheader("Cardinality Preview")

cardinality_metrics = st.columns(3)

cardinality_metrics[0].metric(
    "Row Groups",
    f"{cardinality.row_group_count:,}",
)

cardinality_metrics[1].metric(
    "Column Groups",
    f"{cardinality.column_group_count:,}",
)

cardinality_metrics[2].metric(
    "Potential Cells",
    f"{cardinality.potential_cell_count:,}",
)


if cardinality.potential_cell_count > cardinality.max_cells:
    st.error(
        "This pivot configuration would create up to "
        f"{cardinality.potential_cell_count:,} cells, "
        f"which exceeds the configured limit of "
        f"{cardinality.max_cells:,}. "
        "Reduce dimension cardinality or raise the safety limit."
    )
    st.stop()


if cardinality.potential_cell_count > cardinality.max_cells * 0.75:
    st.warning(
        "This configuration is close to the current pivot-size limit. "
        "The resulting matrix may be wide or expensive to inspect."
    )


try:
    result = build_pivot_table(
        working_data,
        row_dimensions,
        column_dimensions,
        value_column=value_column,
        aggregation=aggregation,
        include_missing_dimensions=(include_missing_dimensions),
        max_cells=int(max_cells),
    )

except (
    TypeError,
    ValueError,
) as exc:
    st.error(str(exc))
    st.stop()


st.divider()

st.subheader("Analysis Summary")

summary_metrics = st.columns(4)

summary_metrics[0].metric(
    "Source Rows",
    f"{result.source_row_count:,}",
)

summary_metrics[1].metric(
    "Included Rows",
    f"{result.included_row_count:,}",
)

summary_metrics[2].metric(
    "Excluded Rows",
    f"{result.excluded_dimension_row_count:,}",
)

summary_metrics[3].metric(
    "Observed Groups",
    f"{result.group_count:,}",
)


configuration_frame = pd.DataFrame(
    [
        {
            "Setting": "Row dimensions",
            "Value": ", ".join(result.row_dimensions),
        },
        {
            "Setting": "Column dimensions",
            "Value": (", ".join(result.column_dimensions) if result.column_dimensions else "None"),
        },
        {
            "Setting": "Value column",
            "Value": (result.value_column if result.value_column is not None else "None"),
        },
        {
            "Setting": "Aggregation",
            "Value": aggregation_label(result.aggregation),
        },
        {
            "Setting": "Missing dimensions",
            "Value": ("Included" if result.include_missing_dimensions else "Excluded"),
        },
    ]
)

with st.expander("View analysis configuration"):
    st.dataframe(
        configuration_frame,
        hide_index=True,
        width="stretch",
    )


matrix_tab, tidy_tab = st.tabs(
    [
        "Pivot Table",
        "Grouped Data",
    ]
)


with matrix_tab:
    st.subheader("Pivot Table")

    if result.table.empty:
        st.info("The selected configuration produced no pivoted observations.")

    else:
        st.dataframe(
            result.table,
            width="stretch",
        )

        st.caption(
            "Missing cells represent dimension combinations that were "
            "not observed or produced no aggregated value. "
            "They are not automatically filled with zero."
        )


with tidy_tab:
    st.subheader("Grouped Data")

    if result.tidy.empty:
        st.info("The selected configuration produced no grouped observations.")

    else:
        st.dataframe(
            result.tidy,
            hide_index=True,
            width="stretch",
        )

        st.caption(
            "This view contains one row per observed dimension combination "
            "and is often easier to export or use in downstream analysis."
        )
