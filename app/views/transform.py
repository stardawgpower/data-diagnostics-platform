import pandas as pd
import streamlit as st

from data_diagnostics.analysis import profile_dataset
from data_diagnostics.quality import QualityIssueCode
from data_diagnostics.transformation import (
    DropColumns,
    RemoveEmptyRows,
    ReplaceValueWithMissing,
    TransformationError,
    TransformationPlan,
    apply_transformation_plan,
)


def build_sentinel_table(profile) -> pd.DataFrame:
    """Build an editable table of possible sentinel findings."""

    rows = []

    for issue in profile.quality.issues:
        if issue.code != QualityIssueCode.POSSIBLE_SENTINEL:
            continue

        rows.append(
            {
                "Apply": True,
                "Column": issue.column,
                "Value": issue.observed_value,
                "Occurrences": (issue.affected_count or 0),
                "Affected %": (
                    f"{issue.affected_ratio:.1%}" if issue.affected_ratio is not None else "—"
                ),
            }
        )

    return pd.DataFrame(rows)


def structural_empty_columns(
    profile,
) -> list[str]:
    """Return columns flagged as structural empty columns."""

    return [
        issue.column
        for issue in profile.quality.issues
        if (issue.code == QualityIssueCode.STRUCTURAL_EMPTY_COLUMN and issue.column is not None)
    ]


def build_operations(
    edited_sentinels: pd.DataFrame,
    remove_empty_rows: bool,
    drop_columns: list[str],
):
    """Create ordered transformation operations."""

    operations = []

    if not edited_sentinels.empty:
        selected = edited_sentinels[edited_sentinels["Apply"]]

        sentinel_groups: dict[
            object,
            list[str],
        ] = {}

        for _, row in selected.iterrows():
            value = row["Value"]
            column = str(row["Column"])

            sentinel_groups.setdefault(
                value,
                [],
            ).append(column)

        for value, columns in sentinel_groups.items():
            operations.append(
                ReplaceValueWithMissing(
                    columns=tuple(columns),
                    value=value,
                )
            )

    if remove_empty_rows:
        operations.append(RemoveEmptyRows())

    if drop_columns:
        unique_columns = tuple(dict.fromkeys(drop_columns))

        operations.append(
            DropColumns(
                columns=unique_columns,
            )
        )

    return tuple(operations)


st.title("Transformation Workspace")

st.write("Review data-quality findings and build an explicit, reproducible working dataset.")

st.caption(
    "Every plan is reapplied from the immutable raw dataset. The original upload is never modified."
)


raw_data = st.session_state.raw_data
raw_profile = st.session_state.raw_profile


if raw_data is None:
    st.warning("No dataset is loaded. Open Workspace and upload a dataset first.")
    st.stop()


if raw_profile is None:
    raw_profile = profile_dataset(raw_data)


structural_columns = structural_empty_columns(raw_profile)

sentinel_table = build_sentinel_table(raw_profile)


st.subheader("Suggested Transformations")

with st.form(
    "transformation_plan_form",
    border=True,
):
    st.markdown("#### 1. Sentinel values")

    if sentinel_table.empty:
        st.success("No possible sentinel-value patterns were detected.")

        edited_sentinels = sentinel_table

    else:
        st.caption("Select only values you have reviewed and want to interpret as missing.")

        edited_sentinels = st.data_editor(
            sentinel_table,
            hide_index=True,
            width="stretch",
            num_rows="fixed",
            disabled=[
                "Column",
                "Value",
                "Occurrences",
                "Affected %",
            ],
            column_config={
                "Apply": (
                    st.column_config.CheckboxColumn(
                        "Apply",
                        help=("Replace this value with missing in the selected column."),
                    )
                ),
            },
            key="sentinel_editor",
        )

    st.divider()

    st.markdown("#### 2. Completely empty rows")

    empty_row_count = raw_profile.quality.all_missing_row_count

    remove_empty_rows = st.checkbox(
        (f"Remove completely empty rows ({empty_row_count:,} detected)"),
        value=empty_row_count > 0,
        disabled=empty_row_count == 0,
    )

    st.divider()

    st.markdown("#### 3. Columns")

    selected_structural_columns = st.multiselect(
        "Structural empty columns to remove",
        options=structural_columns,
        default=structural_columns,
        help=(
            "These columns appear to be empty fields "
            "created by file structure such as trailing delimiters."
        ),
    )

    other_column_options = [
        str(column) for column in raw_data.columns if str(column) not in selected_structural_columns
    ]

    additional_drop_columns = st.multiselect(
        "Additional columns to remove",
        options=other_column_options,
        default=[],
        help=("Optional. These columns are removed only from the working dataset."),
    )

    st.divider()

    st.caption("Execution order: sentinel replacement → empty-row removal → column removal.")

    apply_plan = st.form_submit_button(
        "Apply transformations",
        type="primary",
        width="stretch",
    )


if apply_plan:
    columns_to_drop = [
        *selected_structural_columns,
        *additional_drop_columns,
    ]

    operations = build_operations(
        edited_sentinels=edited_sentinels,
        remove_empty_rows=remove_empty_rows,
        drop_columns=columns_to_drop,
    )

    plan = TransformationPlan(
        operations=operations,
    )

    try:
        with st.spinner("Applying transformation plan..."):
            result = apply_transformation_plan(
                raw_data,
                plan,
            )

            working_profile = profile_dataset(result.data)

    except TransformationError as exc:
        st.error(str(exc))

    else:
        st.session_state.transformation_plan = plan

        st.session_state.transformation_result = result

        st.session_state.working_data = result.data

        st.session_state.working_profile = working_profile

        st.success("Transformation plan applied successfully.")


st.divider()

st.subheader("Working Dataset")


result = st.session_state.transformation_result

working_data = st.session_state.working_data

working_profile = st.session_state.working_profile


if working_data is None:
    working_data = raw_data


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Raw Rows",
    f"{len(raw_data):,}",
)

metric_columns[1].metric(
    "Working Rows",
    f"{len(working_data):,}",
    delta=(len(working_data) - len(raw_data)),
)

metric_columns[2].metric(
    "Raw Columns",
    f"{len(raw_data.columns):,}",
)

metric_columns[3].metric(
    "Working Columns",
    f"{len(working_data.columns):,}",
    delta=(len(working_data.columns) - len(raw_data.columns)),
)


if result is not None:
    with st.status(
        "Latest transformation plan",
        state="complete",
        expanded=True,
    ):
        for index, entry in enumerate(
            result.log,
            start=1,
        ):
            st.markdown(f"**{index}. {entry.operation}**")

            st.write(entry.message)

            st.caption(
                f"Affected rows: {entry.affected_rows:,} · Affected cells: {entry.affected_cells:,}"
            )

    if working_profile is not None:
        comparison_columns = st.columns(3)

        comparison_columns[0].metric(
            "Missing Cells",
            f"{working_profile.quality.missing_cell_count:,}",
            delta=(
                working_profile.quality.missing_cell_count - raw_profile.quality.missing_cell_count
            ),
        )

        comparison_columns[1].metric(
            "Empty Rows",
            f"{working_profile.quality.all_missing_row_count:,}",
            delta=(
                working_profile.quality.all_missing_row_count
                - raw_profile.quality.all_missing_row_count
            ),
        )

        comparison_columns[2].metric(
            "Quality Findings",
            working_profile.quality.issue_count,
            delta=(working_profile.quality.issue_count - raw_profile.quality.issue_count),
        )


st.dataframe(
    working_data.head(20),
    width="stretch",
)


if st.button(
    "Reset working dataset",
    type="secondary",
):
    st.session_state.working_data = raw_data.copy(deep=True)

    st.session_state.working_profile = raw_profile

    st.session_state.transformation_plan = None
    st.session_state.transformation_result = None

    st.rerun()
