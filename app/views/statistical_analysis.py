import math

import pandas as pd
import streamlit as st

from data_diagnostics.analysis import (
    MultiGroupMethod,
    TwoGroupMethod,
    analyze_categorical_association,
    compare_multiple_numeric_groups,
    compare_two_numeric_groups,
)
from data_diagnostics.ingestion import SemanticType


def numeric_columns_from_profile(profile) -> list[str]:
    """Return numeric columns suitable for statistical comparison."""

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


def observed_values(
    data: pd.DataFrame,
    column: str,
) -> list[object]:
    """Return observed non-missing values in first-seen order."""

    values = data[column].dropna()

    try:
        return values.drop_duplicates().tolist()
    except TypeError:
        return []


def group_value_label(value: object) -> str:
    """Return a readable group-value label."""

    return str(value)


def two_group_method_label(
    method: TwoGroupMethod,
) -> str:
    """Return a readable two-group method label."""

    labels = {
        TwoGroupMethod.WELCH_T: "Welch's t-test",
        TwoGroupMethod.MANN_WHITNEY: "Mann–Whitney U",
    }

    return labels[method]


def multi_group_method_label(
    method: MultiGroupMethod,
) -> str:
    """Return a readable multi-group method label."""

    labels = {
        MultiGroupMethod.ONE_WAY_ANOVA: "One-way ANOVA",
        MultiGroupMethod.KRUSKAL_WALLIS: "Kruskal–Wallis",
    }

    return labels[method]


def effect_size_label(name: str) -> str:
    """Return a readable effect-size label."""

    labels = {
        "hedges_g": "Hedges' g",
        "rank_biserial_correlation": "Rank-biserial correlation",
        "eta_squared": "Eta-squared",
        "epsilon_squared": "Epsilon-squared",
    }

    return labels.get(
        name,
        name.replace("_", " ").title(),
    )


def format_number(
    value: float | None,
    *,
    digits: int = 6,
) -> str:
    """Format an optional floating-point value."""

    if value is None:
        return "Unavailable"

    if not math.isfinite(value):
        return str(value)

    return f"{value:.{digits}g}"


def threshold_label(
    significant: bool | None,
) -> str:
    """Describe the p-value comparison without causal interpretation."""

    if significant is None:
        return "Unavailable"

    if significant:
        return "p < α"

    return "p ≥ α"


def numeric_group_summary_frame(
    summaries,
) -> pd.DataFrame:
    """Convert numeric group summaries to a display DataFrame."""

    return pd.DataFrame(
        [
            {
                "Group": summary.group,
                "Count": summary.count,
                "Mean": summary.mean,
                "Median": summary.median,
                "Std. Dev.": summary.standard_deviation,
                "Minimum": summary.minimum,
                "Maximum": summary.maximum,
            }
            for summary in summaries
        ]
    )


st.title("Statistical Analysis")

st.write(
    "Run explicit inferential statistical tests on the working dataset "
    "with transparent observation counts, test statistics, p-values, "
    "and effect sizes."
)

st.caption(
    "The workspace does not automatically choose a statistical test, "
    "impute missing values, transform the dataset, or interpret a "
    "p-value as proof of a hypothesis."
)


working_data = st.session_state.working_data
working_profile = st.session_state.working_profile

dataset_fingerprint = st.session_state.get("dataset_fingerprint") or "dataset"

widget_prefix = f"statistical_analysis::{dataset_fingerprint}"


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
    st.info("The working dataset has no columns available for statistical analysis.")
    st.stop()


two_group_tab, multi_group_tab, categorical_tab = st.tabs(
    [
        "Two Groups",
        "Multiple Groups",
        "Categorical Association",
    ]
)


with two_group_tab:
    st.subheader("Two Independent Numeric Groups")

    st.write(
        "Compare a numeric measure between two explicitly selected "
        "groups using either Welch's t-test or Mann–Whitney U."
    )

    if not numeric_columns:
        st.info("No numeric continuous or discrete columns are available for two-group analysis.")

    else:
        value_column = st.selectbox(
            "Numeric value column",
            options=numeric_columns,
            format_func=lambda column: column_label(
                column,
                semantic_types,
            ),
            key=f"{widget_prefix}::two::value_column",
        )

        group_columns = [column for column in all_columns if column != value_column]

        if not group_columns:
            st.info("A second column is required to define groups.")

        else:
            group_column = st.selectbox(
                "Group column",
                options=group_columns,
                format_func=lambda column: column_label(
                    column,
                    semantic_types,
                ),
                key=(f"{widget_prefix}::two::group_column::{value_column}"),
            )

            group_values = observed_values(
                working_data,
                group_column,
            )

            if len(group_values) < 2:
                st.info(
                    "The selected group column must contain at least "
                    "two observed non-missing values."
                )

            else:
                if len(group_values) > 50:
                    st.warning(
                        f"The selected group column has "
                        f"{len(group_values):,} observed values. "
                        "Choose groups carefully; a categorical or "
                        "discrete grouping column is usually easier "
                        "to interpret."
                    )

                selector_columns = st.columns(2)

                with selector_columns[0]:
                    group_a = st.selectbox(
                        "Group A",
                        options=group_values,
                        format_func=group_value_label,
                        key=(f"{widget_prefix}::two::group_a::{group_column}"),
                    )

                with selector_columns[1]:
                    group_b = st.selectbox(
                        "Group B",
                        options=group_values,
                        index=1,
                        format_func=group_value_label,
                        key=(f"{widget_prefix}::two::group_b::{group_column}"),
                    )

                method = st.selectbox(
                    "Method",
                    options=[
                        TwoGroupMethod.WELCH_T,
                        TwoGroupMethod.MANN_WHITNEY,
                    ],
                    format_func=two_group_method_label,
                    key=f"{widget_prefix}::two::method",
                )

                alpha = st.number_input(
                    "Alpha (α)",
                    min_value=0.001,
                    max_value=0.500,
                    value=0.050,
                    step=0.001,
                    format="%.3f",
                    help=(
                        "Threshold used only for comparing the returned "
                        "p-value with α. It does not measure effect size "
                        "or practical importance."
                    ),
                    key=f"{widget_prefix}::two::alpha",
                )

                if method == TwoGroupMethod.WELCH_T:
                    st.caption(
                        "Welch's t-test compares group means for "
                        "independent observations and does not assume "
                        "equal group variances."
                    )

                else:
                    st.caption(
                        "Mann–Whitney U compares the rank distributions "
                        "of two independent groups. Interpreting it purely "
                        "as a median test requires stronger distributional "
                        "assumptions."
                    )

                if group_a == group_b:
                    st.info("Choose two different group values.")

                else:
                    try:
                        result = compare_two_numeric_groups(
                            working_data,
                            value_column,
                            group_column,
                            group_a,
                            group_b,
                            method=method,
                            alpha=float(alpha),
                        )

                    except (
                        TypeError,
                        ValueError,
                    ) as exc:
                        st.error(str(exc))

                    else:
                        st.divider()

                        st.subheader("Result")

                        metrics = st.columns(5)

                        metrics[0].metric(
                            "Selected Rows",
                            f"{result.selected_row_count:,}",
                        )

                        metrics[1].metric(
                            "Used Rows",
                            f"{result.used_row_count:,}",
                        )

                        metrics[2].metric(
                            "Excluded Values",
                            f"{result.excluded_value_row_count:,}",
                        )

                        metrics[3].metric(
                            "p-value",
                            format_number(result.p_value),
                        )

                        metrics[4].metric(
                            "Threshold",
                            threshold_label(result.significant),
                        )

                        result_frame = pd.DataFrame(
                            [
                                {
                                    "Measure": "Method",
                                    "Value": two_group_method_label(result.method),
                                },
                                {
                                    "Measure": "Statistic",
                                    "Value": format_number(result.statistic),
                                },
                                {
                                    "Measure": "p-value",
                                    "Value": format_number(result.p_value),
                                },
                                {
                                    "Measure": "Alpha",
                                    "Value": format_number(result.alpha),
                                },
                                {
                                    "Measure": "Effect size",
                                    "Value": effect_size_label(result.effect_size_name),
                                },
                                {
                                    "Measure": "Effect size value",
                                    "Value": format_number(result.effect_size),
                                },
                            ]
                        )

                        st.dataframe(
                            result_frame,
                            hide_index=True,
                            width="stretch",
                        )

                        st.subheader("Group Descriptives")

                        summaries = (
                            result.group_a_summary,
                            result.group_b_summary,
                        )

                        st.dataframe(
                            numeric_group_summary_frame(summaries),
                            hide_index=True,
                            width="stretch",
                        )

                        st.caption(
                            "Rows from unselected groups are outside this "
                            "comparison. Within the selected groups, missing "
                            "and non-finite numeric values are excluded."
                        )


with multi_group_tab:
    st.subheader("Multiple Independent Numeric Groups")

    st.write(
        "Compare a numeric measure across explicitly selected groups "
        "using one-way ANOVA or Kruskal–Wallis."
    )

    if not numeric_columns:
        st.info("No numeric continuous or discrete columns are available for multi-group analysis.")

    else:
        multi_value_column = st.selectbox(
            "Numeric value column",
            options=numeric_columns,
            format_func=lambda column: column_label(
                column,
                semantic_types,
            ),
            key=f"{widget_prefix}::multi::value_column",
        )

        multi_group_columns = [column for column in all_columns if column != multi_value_column]

        if not multi_group_columns:
            st.info("A second column is required to define groups.")

        else:
            multi_group_column = st.selectbox(
                "Group column",
                options=multi_group_columns,
                format_func=lambda column: column_label(
                    column,
                    semantic_types,
                ),
                key=(f"{widget_prefix}::multi::group_column::{multi_value_column}"),
            )

            multi_group_values = observed_values(
                working_data,
                multi_group_column,
            )

            if len(multi_group_values) < 2:
                st.info(
                    "The selected group column must contain at least "
                    "two observed non-missing values."
                )

            else:
                if len(multi_group_values) > 50:
                    st.warning(
                        f"The selected group column has "
                        f"{len(multi_group_values):,} observed values. "
                        "Select only groups that belong in the intended "
                        "comparison."
                    )

                default_groups = (
                    multi_group_values[:3] if len(multi_group_values) >= 3 else multi_group_values
                )

                selected_groups = st.multiselect(
                    "Groups to compare",
                    options=multi_group_values,
                    default=default_groups,
                    format_func=group_value_label,
                    help=("Only explicitly selected groups are included in the statistical test."),
                    key=(f"{widget_prefix}::multi::groups::{multi_group_column}"),
                )

                multi_method = st.selectbox(
                    "Method",
                    options=[
                        MultiGroupMethod.ONE_WAY_ANOVA,
                        MultiGroupMethod.KRUSKAL_WALLIS,
                    ],
                    format_func=multi_group_method_label,
                    key=f"{widget_prefix}::multi::method",
                )

                multi_alpha = st.number_input(
                    "Alpha (α)",
                    min_value=0.001,
                    max_value=0.500,
                    value=0.050,
                    step=0.001,
                    format="%.3f",
                    help=("Threshold used to compare the returned p-value with α."),
                    key=f"{widget_prefix}::multi::alpha",
                )

                if multi_method == MultiGroupMethod.ONE_WAY_ANOVA:
                    st.caption(
                        "One-way ANOVA compares means across independent "
                        "groups and assumes approximately normal residuals "
                        "and equal population variances."
                    )

                else:
                    st.caption(
                        "Kruskal–Wallis compares rank distributions across "
                        "independent groups. A significant result does not "
                        "identify which specific groups differ."
                    )

                if len(selected_groups) < 2:
                    st.info("Select at least two groups.")

                else:
                    try:
                        multi_result = compare_multiple_numeric_groups(
                            working_data,
                            multi_value_column,
                            multi_group_column,
                            selected_groups,
                            method=multi_method,
                            alpha=float(multi_alpha),
                        )

                    except (
                        TypeError,
                        ValueError,
                    ) as exc:
                        st.error(str(exc))

                    else:
                        st.divider()

                        st.subheader("Result")

                        multi_metrics = st.columns(5)

                        multi_metrics[0].metric(
                            "Selected Rows",
                            f"{multi_result.selected_row_count:,}",
                        )

                        multi_metrics[1].metric(
                            "Used Rows",
                            f"{multi_result.used_row_count:,}",
                        )

                        multi_metrics[2].metric(
                            "Excluded Values",
                            (f"{multi_result.excluded_value_row_count:,}"),
                        )

                        multi_metrics[3].metric(
                            "p-value",
                            format_number(multi_result.p_value),
                        )

                        multi_metrics[4].metric(
                            "Threshold",
                            threshold_label(multi_result.significant),
                        )

                        multi_result_frame = pd.DataFrame(
                            [
                                {
                                    "Measure": "Method",
                                    "Value": multi_group_method_label(multi_result.method),
                                },
                                {
                                    "Measure": "Statistic",
                                    "Value": format_number(multi_result.statistic),
                                },
                                {
                                    "Measure": "p-value",
                                    "Value": format_number(multi_result.p_value),
                                },
                                {
                                    "Measure": "Alpha",
                                    "Value": format_number(multi_result.alpha),
                                },
                                {
                                    "Measure": "Effect size",
                                    "Value": effect_size_label(multi_result.effect_size_name),
                                },
                                {
                                    "Measure": "Effect size value",
                                    "Value": format_number(multi_result.effect_size),
                                },
                            ]
                        )

                        st.dataframe(
                            multi_result_frame,
                            hide_index=True,
                            width="stretch",
                        )

                        st.subheader("Group Descriptives")

                        st.dataframe(
                            numeric_group_summary_frame(multi_result.group_summaries),
                            hide_index=True,
                            width="stretch",
                        )

                        st.caption(
                            "The omnibus test evaluates the selected groups "
                            "jointly. This workspace does not automatically "
                            "run post-hoc pairwise tests or apply multiple-"
                            "comparison corrections."
                        )


with categorical_tab:
    st.subheader("Categorical Association")

    st.write(
        "Evaluate association between two categorical or discrete "
        "variables using Pearson's chi-square test."
    )

    if len(all_columns) < 2:
        st.info("At least two columns are required for categorical association analysis.")

    else:
        categorical_row_column = st.selectbox(
            "Row variable",
            options=all_columns,
            format_func=lambda column: column_label(
                column,
                semantic_types,
            ),
            key=f"{widget_prefix}::categorical::row_column",
        )

        categorical_column_options = [
            column for column in all_columns if column != categorical_row_column
        ]

        categorical_column_column = st.selectbox(
            "Column variable",
            options=categorical_column_options,
            format_func=lambda column: column_label(
                column,
                semantic_types,
            ),
            key=(f"{widget_prefix}::categorical::column_column::{categorical_row_column}"),
        )

        categorical_options = st.columns(2)

        with categorical_options[0]:
            categorical_alpha = st.number_input(
                "Alpha (α)",
                min_value=0.001,
                max_value=0.500,
                value=0.050,
                step=0.001,
                format="%.3f",
                key=f"{widget_prefix}::categorical::alpha",
            )

        with categorical_options[1]:
            max_cells = st.number_input(
                "Maximum contingency cells",
                min_value=4,
                max_value=10_000,
                value=400,
                step=10,
                help=(
                    "Safety guard against accidentally constructing a "
                    "very large contingency table from high-cardinality "
                    "columns."
                ),
                key=f"{widget_prefix}::categorical::max_cells",
            )

        st.caption(
            "Rows missing either selected category are excluded. "
            "Pearson's chi-square test is computed without Yates "
            "continuity correction."
        )

        try:
            categorical_result = analyze_categorical_association(
                working_data,
                categorical_row_column,
                categorical_column_column,
                alpha=float(categorical_alpha),
                max_cells=int(max_cells),
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            st.error(str(exc))

        else:
            st.divider()

            st.subheader("Result")

            categorical_metrics = st.columns(5)

            categorical_metrics[0].metric(
                "Used Rows",
                f"{categorical_result.used_row_count:,}",
            )

            categorical_metrics[1].metric(
                "Excluded Missing",
                (f"{categorical_result.excluded_missing_row_count:,}"),
            )

            categorical_metrics[2].metric(
                "Contingency Cells",
                f"{categorical_result.cell_count:,}",
            )

            categorical_metrics[3].metric(
                "p-value",
                format_number(categorical_result.p_value),
            )

            categorical_metrics[4].metric(
                "Threshold",
                threshold_label(categorical_result.significant),
            )

            association_frame = pd.DataFrame(
                [
                    {
                        "Measure": "Chi-square statistic",
                        "Value": format_number(categorical_result.statistic),
                    },
                    {
                        "Measure": "Degrees of freedom",
                        "Value": str(categorical_result.degrees_of_freedom),
                    },
                    {
                        "Measure": "p-value",
                        "Value": format_number(categorical_result.p_value),
                    },
                    {
                        "Measure": "Alpha",
                        "Value": format_number(categorical_result.alpha),
                    },
                    {
                        "Measure": "Cramér's V",
                        "Value": format_number(categorical_result.cramers_v),
                    },
                    {
                        "Measure": "Minimum expected frequency",
                        "Value": format_number(categorical_result.minimum_expected_frequency),
                    },
                    {
                        "Measure": "Expected cells below 5",
                        "Value": (f"{categorical_result.expected_below_five_count:,}"),
                    },
                    {
                        "Measure": "Share of expected cells below 5",
                        "Value": (f"{categorical_result.expected_below_five_ratio:.1%}"),
                    },
                ]
            )

            st.dataframe(
                association_frame,
                hide_index=True,
                width="stretch",
            )

            if (
                categorical_result.minimum_expected_frequency < 1.0
                or categorical_result.expected_below_five_ratio > 0.20
            ):
                st.warning(
                    "The expected-frequency table is sparse. "
                    "A common chi-square approximation guideline is to "
                    "avoid expected counts below 1 and to keep the share "
                    "of expected cells below 5 relatively small. "
                    "Interpret the asymptotic p-value cautiously."
                )

            observed_tab, expected_tab = st.tabs(
                [
                    "Observed Counts",
                    "Expected Counts",
                ]
            )

            with observed_tab:
                st.dataframe(
                    categorical_result.observed,
                    width="stretch",
                )

            with expected_tab:
                st.dataframe(
                    categorical_result.expected,
                    width="stretch",
                )

            st.caption(
                "Cramér's V is reported as an association effect-size "
                "measure. This analysis does not establish causation."
            )
