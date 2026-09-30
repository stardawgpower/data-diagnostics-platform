import pandas as pd
import streamlit as st

from data_diagnostics.analysis import (
    prepare_numeric_pair,
    profile_dataset,
    summarize_categorical_frequencies,
    summarize_missingness,
    summarize_numeric_distribution,
    summarize_numeric_outliers,
)
from data_diagnostics.ingestion import SemanticType


def format_number(
    value: float | None,
    *,
    decimals: int = 3,
) -> str:
    """Format an optional numeric value for display."""

    if value is None:
        return "—"

    return f"{value:,.{decimals}f}"


def numeric_columns_from_profile(profile) -> list[str]:
    """Return columns classified as numeric for exploratory analysis."""

    numeric_types = {
        SemanticType.NUMERIC_DISCRETE,
        SemanticType.NUMERIC_CONTINUOUS,
    }

    return [
        column.schema.name
        for column in profile.columns
        if column.schema.semantic_type in numeric_types
    ]


def categorical_columns_from_profile(profile) -> list[str]:
    """Return categorical-like columns suitable for frequency analysis."""

    categorical_types = {
        SemanticType.BINARY,
        SemanticType.CATEGORICAL,
        SemanticType.TEXT,
    }

    return [
        column.schema.name
        for column in profile.columns
        if column.schema.semantic_type in categorical_types
    ]


st.title("Explore")

st.write(
    "Explore distributions, missingness, categorical frequencies, "
    "potential numeric outliers, and numeric pairs."
)

st.caption(
    "Exploration uses the current working dataset. "
    "Charts and summaries are descriptive only and do not modify data."
)


working_data = st.session_state.working_data
working_profile = st.session_state.working_profile


if working_data is None:
    st.warning("No dataset is loaded. Open Workspace and upload a dataset first.")
    st.stop()


if working_profile is None:
    working_profile = profile_dataset(
        working_data,
    )


numeric_columns = numeric_columns_from_profile(
    working_profile,
)

categorical_columns = categorical_columns_from_profile(
    working_profile,
)


overview_tab, numeric_tab, categorical_tab, pair_tab = st.tabs(
    [
        "Overview",
        "Numeric",
        "Categorical",
        "Numeric Pair",
    ]
)


with overview_tab:
    st.subheader("Working Dataset Overview")

    overview_metrics = st.columns(4)

    overview_metrics[0].metric(
        "Rows",
        f"{len(working_data):,}",
    )

    overview_metrics[1].metric(
        "Columns",
        f"{len(working_data.columns):,}",
    )

    overview_metrics[2].metric(
        "Numeric Columns",
        f"{len(numeric_columns):,}",
    )

    overview_metrics[3].metric(
        "Categorical Columns",
        f"{len(categorical_columns):,}",
    )

    st.divider()

    st.subheader("Missingness")

    missingness = summarize_missingness(
        working_data,
    )

    missingness_frame = pd.DataFrame(
        [
            {
                "Column": summary.column,
                "Missing": summary.missing_count,
                "Missing %": summary.missing_ratio,
            }
            for summary in missingness
        ]
    )

    missing_columns = missingness_frame[missingness_frame["Missing"] > 0].copy()

    if missing_columns.empty:
        st.success("No missing values are present in the working dataset.")

    else:
        missing_columns = missing_columns.sort_values(
            by=[
                "Missing",
                "Column",
            ],
            ascending=[
                False,
                True,
            ],
        )

        chart_data = missing_columns[
            [
                "Column",
                "Missing",
            ]
        ].set_index("Column")

        st.bar_chart(
            chart_data,
            width="stretch",
        )

        display_frame = missing_columns.copy()

        display_frame["Missing %"] = display_frame["Missing %"] * 100

        st.dataframe(
            display_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Missing %": st.column_config.NumberColumn(
                    "Missing %",
                    format="%.2f%%",
                ),
            },
        )


with numeric_tab:
    st.subheader("Numeric Distribution")

    if not numeric_columns:
        st.info("No numeric columns are available in the working dataset.")

    else:
        selected_numeric_column = st.selectbox(
            "Numeric column",
            options=numeric_columns,
            key="explore_numeric_column",
        )

        histogram_bins = st.slider(
            "Histogram bins",
            min_value=5,
            max_value=100,
            value=20,
            step=5,
            key="explore_histogram_bins",
        )

        distribution = summarize_numeric_distribution(
            working_data,
            selected_numeric_column,
            bins=histogram_bins,
        )

        metric_columns = st.columns(5)

        metric_columns[0].metric(
            "Valid",
            f"{distribution.valid_count:,}",
        )

        metric_columns[1].metric(
            "Missing",
            f"{distribution.missing_count:,}",
        )

        metric_columns[2].metric(
            "Non-finite",
            f"{distribution.non_finite_count:,}",
        )

        metric_columns[3].metric(
            "Mean",
            format_number(
                distribution.mean,
            ),
        )

        metric_columns[4].metric(
            "Median",
            format_number(
                distribution.median,
            ),
        )

        statistics_frame = pd.DataFrame(
            [
                {
                    "Statistic": "Minimum",
                    "Value": distribution.minimum,
                },
                {
                    "Statistic": "Q1",
                    "Value": distribution.first_quartile,
                },
                {
                    "Statistic": "Median",
                    "Value": distribution.median,
                },
                {
                    "Statistic": "Q3",
                    "Value": distribution.third_quartile,
                },
                {
                    "Statistic": "Maximum",
                    "Value": distribution.maximum,
                },
                {
                    "Statistic": "Standard deviation",
                    "Value": distribution.standard_deviation,
                },
            ]
        )

        statistics_column, histogram_column = st.columns(
            [
                1,
                2,
            ]
        )

        with statistics_column:
            st.markdown("#### Distribution Statistics")

            st.dataframe(
                statistics_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Value": st.column_config.NumberColumn(
                        "Value",
                        format="%.4f",
                    ),
                },
            )

        with histogram_column:
            st.markdown("#### Histogram")

            if not distribution.histogram:
                st.info("No finite numeric values are available for histogram construction.")

            else:
                histogram_frame = pd.DataFrame(
                    [
                        {
                            "Bin": (f"{histogram_bin.left:.4g} – {histogram_bin.right:.4g}"),
                            "Count": histogram_bin.count,
                        }
                        for histogram_bin in distribution.histogram
                    ]
                )

                st.bar_chart(
                    histogram_frame,
                    x="Bin",
                    y="Count",
                    width="stretch",
                )

        st.divider()

        st.subheader("Potential Outliers")

        st.caption(
            "Potential outliers are identified using the IQR rule. "
            "They are diagnostic findings only and are not removed."
        )

        iqr_multiplier = st.number_input(
            "IQR multiplier",
            min_value=0.1,
            max_value=10.0,
            value=1.5,
            step=0.1,
            key="explore_iqr_multiplier",
        )

        outliers = summarize_numeric_outliers(
            working_data,
            selected_numeric_column,
            multiplier=iqr_multiplier,
        )

        outlier_metrics = st.columns(4)

        outlier_metrics[0].metric(
            "Potential Outliers",
            f"{outliers.outlier_count:,}",
        )

        outlier_metrics[1].metric(
            "Lower",
            f"{outliers.lower_outlier_count:,}",
        )

        outlier_metrics[2].metric(
            "Upper",
            f"{outliers.upper_outlier_count:,}",
        )

        outlier_metrics[3].metric(
            "Outlier %",
            (f"{outliers.outlier_ratio:.2%}" if outliers.outlier_ratio is not None else "—"),
        )

        bounds_frame = pd.DataFrame(
            [
                {
                    "Statistic": "Q1",
                    "Value": outliers.first_quartile,
                },
                {
                    "Statistic": "Q3",
                    "Value": outliers.third_quartile,
                },
                {
                    "Statistic": "IQR",
                    "Value": outliers.iqr,
                },
                {
                    "Statistic": "Lower bound",
                    "Value": outliers.lower_bound,
                },
                {
                    "Statistic": "Upper bound",
                    "Value": outliers.upper_bound,
                },
            ]
        )

        st.dataframe(
            bounds_frame,
            hide_index=True,
            width="stretch",
            column_config={
                "Value": st.column_config.NumberColumn(
                    "Value",
                    format="%.4f",
                ),
            },
        )


with categorical_tab:
    st.subheader("Categorical Frequencies")

    if not categorical_columns:
        st.info("No categorical, binary, or text columns are available for frequency analysis.")

    else:
        selected_categorical_column = st.selectbox(
            "Categorical column",
            options=categorical_columns,
            key="explore_categorical_column",
        )

        top_n = st.slider(
            "Top categories",
            min_value=1,
            max_value=50,
            value=10,
            key="explore_top_categories",
        )

        category_summary = summarize_categorical_frequencies(
            working_data,
            selected_categorical_column,
            top_n=top_n,
        )

        categorical_metrics = st.columns(4)

        categorical_metrics[0].metric(
            "Rows",
            f"{category_summary.total_count:,}",
        )

        categorical_metrics[1].metric(
            "Non-missing",
            f"{category_summary.non_missing_count:,}",
        )

        categorical_metrics[2].metric(
            "Missing",
            f"{category_summary.missing_count:,}",
        )

        categorical_metrics[3].metric(
            "Unique",
            f"{category_summary.unique_count:,}",
        )

        frequency_rows = [
            {
                "Category": str(frequency.value),
                "Count": frequency.count,
                "Share of rows": frequency.ratio,
            }
            for frequency in category_summary.frequencies
        ]

        if category_summary.other_count > 0:
            frequency_rows.append(
                {
                    "Category": "Other categories",
                    "Count": category_summary.other_count,
                    "Share of rows": category_summary.other_ratio,
                }
            )

        frequency_frame = pd.DataFrame(frequency_rows)

        if frequency_frame.empty:
            st.info("No non-missing values are available for frequency analysis.")

        else:
            st.bar_chart(
                frequency_frame,
                x="Category",
                y="Count",
                width="stretch",
            )

            display_frequency_frame = frequency_frame.copy()

            display_frequency_frame["Share of rows"] = (
                display_frequency_frame["Share of rows"] * 100
            )

            st.dataframe(
                display_frequency_frame,
                hide_index=True,
                width="stretch",
                column_config={
                    "Share of rows": st.column_config.NumberColumn(
                        "Share of rows %",
                        format="%.2f%%",
                    ),
                },
            )


with pair_tab:
    st.subheader("Numeric Pair Exploration")

    st.caption(
        "Inspect paired finite numeric observations. "
        "Correlation and statistical relationship analysis "
        "will be handled separately."
    )

    if len(numeric_columns) < 2:
        st.info("At least two numeric columns are required for numeric-pair exploration.")

    else:
        selector_columns = st.columns(2)

        with selector_columns[0]:
            x_column = st.selectbox(
                "X column",
                options=numeric_columns,
                index=0,
                key="explore_pair_x",
            )

        y_options = [column for column in numeric_columns if column != x_column]

        with selector_columns[1]:
            y_column = st.selectbox(
                "Y column",
                options=y_options,
                index=0,
                key="explore_pair_y",
            )

        pair = prepare_numeric_pair(
            working_data,
            x_column,
            y_column,
        )

        pair_metrics = st.columns(3)

        pair_metrics[0].metric(
            "Total Rows",
            f"{pair.total_count:,}",
        )

        pair_metrics[1].metric(
            "Valid Pairs",
            f"{pair.valid_count:,}",
        )

        pair_metrics[2].metric(
            "Excluded Rows",
            f"{pair.excluded_count:,}",
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
                    "5,000 observations for responsiveness. "
                    "Pair counts above use the full dataset."
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
