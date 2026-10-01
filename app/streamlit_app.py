import streamlit as st

st.set_page_config(
    page_title="Data Diagnostics Platform",
    page_icon="📊",
    layout="wide",
)


SESSION_DEFAULTS = {
    "dataset_name": None,
    "dataset_fingerprint": None,
    "raw_data": None,
    "raw_profile": None,
    "working_data": None,
    "working_profile": None,
    "transformation_plan": None,
    "transformation_result": None,
}


for key, default_value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


workspace_page = st.Page(
    "views/workspace.py",
    title="Workspace",
    icon=":material/upload_file:",
    default=True,
)

profile_page = st.Page(
    "views/profile.py",
    title="Profile",
    icon=":material/analytics:",
)

transform_page = st.Page(
    "views/transform.py",
    title="Transform",
    icon=":material/tune:",
)

explore_page = st.Page(
    "views/explore.py",
    title="Explore",
    icon=":material/query_stats:",
)

relationships_page = st.Page(
    "views/relationships.py",
    title="Relationships",
    icon=":material/hub:",
)

time_analysis_page = st.Page(
    "views/time_analysis.py",
    title="Time Analysis",
    icon=":material/schedule:",
)

page = st.navigation(
    [
        workspace_page,
        profile_page,
        transform_page,
        explore_page,
        relationships_page,
        time_analysis_page,
    ],
    position="top",
)


with st.sidebar:
    st.header("Workspace")

    raw_data = st.session_state.raw_data
    working_data = st.session_state.working_data

    if raw_data is None:
        st.caption("No dataset loaded.")
    else:
        st.markdown(f"**Dataset**  \n{st.session_state.dataset_name}")

        st.divider()

        st.caption("Raw dataset")

        raw_metric_columns = st.columns(2)

        raw_metric_columns[0].metric(
            "Rows",
            f"{len(raw_data):,}",
        )

        raw_metric_columns[1].metric(
            "Columns",
            f"{len(raw_data.columns):,}",
        )

        if working_data is not None:
            st.caption("Working dataset")

            working_metric_columns = st.columns(2)

            working_metric_columns[0].metric(
                "Rows",
                f"{len(working_data):,}",
            )

            working_metric_columns[1].metric(
                "Columns",
                f"{len(working_data.columns):,}",
            )

        transformation_result = st.session_state.transformation_result

        if transformation_result is not None:
            st.caption(
                f"{transformation_result.operation_count} transformation operation(s) applied."
            )


page.run()
