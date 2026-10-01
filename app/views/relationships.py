import pandas as pd
import streamlit as st

from data_diagnostics.analysis import (
    CorrelationMethod,
    RelationshipStatus,
    analyze_numeric_relationship,
    build_correlation_matrix,
    prepare_numeric_pair,
    strongest_correlations,
)
from data_diagnostics.ingestion import SemanticType


def numeric_columns_from_profile(profile) -> list[str]:
    """Return numeric columns available for relationship analysis."""

    numeric_types = {
        SemanticType.NUMERIC_DISCRETE,
        SemanticType.NUMERIC_CONTINUOUS,
    }

    return [
        column.schema.name
        for column in profile.columns
        if column.schema.semantic_type in numeric_types
    ]


def method_label(method: CorrelationMethod) -> str:
    """Return a human-readable correlation method label."""

    labels = {
        CorrelationMethod.PEARSON: "Pearson",
        CorrelationMethod.SPEARMAN: "Spearman",
    }

    return labels[method]


def relationship_status_message(status: RelationshipStatus) -> str:
    """Return explanatory text for an unavailable correlation."""

    messages = {
        RelationshipStatus.INSUFFICIENT_PAIRS: (
            "There are too few paired finite observations to estimate "
            "this relationship at the selected minimum-pair threshold."
        ),
        RelationshipStatus.CONSTANT_INPUT: (
            "Correlation is undefined because at least one selected "
            "column is constant across the valid paired observations."
        ),
    }

    return messages.get(
        status,
        "The correlation estimate is unavailable.",
    )


st.title("Relationships")

st.write("Inspect numeric associations using Pearson or Spearman correlation.")

st.caption(
    "Relationship analysis uses the current working dataset. "
    "Correlation describes association and does not imply causation."
)


working_data = st.session_state.working_data
working_profile = st.session_state.working_profile


if working_data is None:
    st.warning("No dataset is loaded. Open Workspace and upload a dataset first.")
    st.stop()


if working_profile is None:
    st.warning(
        "The working dataset profile is unavailable. Return to Workspace and reload the dataset."
    )
    st.stop()


numeric_columns = numeric_columns_from_profile(
    working_profile,
)


if len(numeric_columns) < 2:
    st.info("At least two numeric columns are required for relationship analysis.")
    st.stop()


configuration_columns = st.columns(
    [
        2,
        1,
    ]
)


with configuration_columns[0]:
    method_label_value = st.radio(
        "Correlation method",
        options=[
            "Pearson",
            "Spearman",
        ],
        horizontal=True,
        help=(
            "Pearson measures linear association. "
            "Spearman measures monotonic association using ranks."
        ),
        key="relationships_method",
    )


method = (
    CorrelationMethod.PEARSON if method_label_value == "Pearson" else CorrelationMethod.SPEARMAN
)


with configuration_columns[1]:
    maximum_minimum_pairs = max(
        2,
        len(working_data),
    )

    minimum_pairs = st.number_input(
        "Minimum valid pairs",
        min_value=2,
        max_value=maximum_minimum_pairs,
        value=min(
            3,
            maximum_minimum_pairs,
        ),
        step=1,
        help=(
            "A correlation is reported only when at least this many "
            "finite paired observations are available."
        ),
        key="relationships_minimum_pairs",
    )


matrix_tab, pair_tab = st.tabs(
    [
        "Correlation Matrix",
        "Pair Inspector",
    ]
)


with matrix_tab:
    st.subheader(f"{method_label(method)} Correlation Matrix")

    default_matrix_columns = numeric_columns if len(numeric_columns) <= 12 else numeric_columns[:12]

    selected_columns = st.multiselect(
        "Columns",
        options=numeric_columns,
        default=default_matrix_columns,
        help=(
            "Select at least two numeric columns. "
            "For wide datasets, the first 12 numeric columns are "
            "selected by default to keep the matrix readable."
        ),
        key="relationships_matrix_columns",
    )

    if len(selected_columns) < 2:
        st.info("Select at least two numeric columns to build the matrix.")

    else:
        matrix = build_correlation_matrix(
            working_data,
            selected_columns,
            method=method,
            minimum_pairs=int(minimum_pairs),
        )

        st.caption(
            "Each coefficient uses only rows where both columns contain "
            "finite numeric values. Undefined coefficients are shown as blank."
        )

        st.markdown("#### Coefficients")

        coefficient_display = matrix.coefficients.round(4).copy()

        st.dataframe(
            coefficient_display,
            width="stretch",
        )

        with st.expander(
            "View pairwise valid-observation counts",
        ):
            st.dataframe(
                matrix.valid_counts,
                width="stretch",
            )

        st.divider()

        st.markdown("#### Largest Absolute Correlations")

        strongest = strongest_correlations(
            matrix,
            top_n=10,
        )

        if not strongest:
            st.info("No valid pairwise correlations are available for the current selection.")

        else:
            strongest_frame = pd.DataFrame(
                [
                    {
                        "X": relationship.x_column,
                        "Y": relationship.y_column,
                        "Correlation": relationship.coefficient,
                        "Absolute correlation": (relationship.absolute_coefficient),
                        "Valid pairs": relationship.valid_count,
                    }
                    for relationship in strongest
                ]
            )

            st.dataframe(
                strongest_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Correlation": st.column_config.NumberColumn(
                        "Correlation",
                        format="%.4f",
                    ),
                    "Absolute correlation": st.column_config.NumberColumn(
                        "Absolute correlation",
                        format="%.4f",
                    ),
                },
            )

            st.caption(
                "Pairs are ordered by absolute correlation magnitude only. "
                "This is not a measure of feature importance or causality."
            )


with pair_tab:
    st.subheader("Pair Inspector")

    selector_columns = st.columns(2)

    with selector_columns[0]:
        x_column = st.selectbox(
            "X column",
            options=numeric_columns,
            index=0,
            key="relationships_pair_x",
        )

    y_options = [column for column in numeric_columns if column != x_column]

    with selector_columns[1]:
        y_column = st.selectbox(
            "Y column",
            options=y_options,
            index=0,
            key="relationships_pair_y",
        )

    relationship = analyze_numeric_relationship(
        working_data,
        x_column,
        y_column,
        method=method,
        minimum_pairs=int(minimum_pairs),
    )

    metric_columns = st.columns(4)

    metric_columns[0].metric(
        "Total Rows",
        f"{relationship.total_count:,}",
    )

    metric_columns[1].metric(
        "Valid Pairs",
        f"{relationship.valid_count:,}",
    )

    metric_columns[2].metric(
        "Excluded Rows",
        f"{relationship.excluded_count:,}",
    )

    metric_columns[3].metric(
        "Correlation",
        (f"{relationship.coefficient:.4f}" if relationship.coefficient is not None else "—"),
    )

    if relationship.status != RelationshipStatus.OK:
        st.info(relationship_status_message(relationship.status))

    pair = prepare_numeric_pair(
        working_data,
        x_column,
        y_column,
    )

    if pair.data.empty:
        st.info("No finite paired observations are available for the selected columns.")

    else:
        chart_data = pair.data

        if len(chart_data) > 5_000:
            chart_data = chart_data.sample(
                n=5_000,
                random_state=42,
            )

            st.caption(
                "Scatterplot displays a reproducible sample of "
                "5,000 paired observations for responsiveness. "
                "Correlation and pair counts use the full valid dataset."
            )

        st.scatter_chart(
            chart_data,
            x=x_column,
            y=y_column,
            width="stretch",
        )

        with st.expander(
            "View paired observations",
        ):
            st.dataframe(
                pair.data.head(1_000),
                width="stretch",
            )

            if len(pair.data) > 1_000:
                st.caption("Preview limited to the first 1,000 valid pairs.")

    st.divider()

    if method == CorrelationMethod.PEARSON:
        st.caption(
            "Pearson correlation summarizes linear association. "
            "A weak Pearson coefficient does not rule out a nonlinear "
            "relationship."
        )

    else:
        st.caption(
            "Spearman correlation summarizes monotonic association "
            "using ranked values and can capture relationships that are "
            "monotonic but not linear."
        )
