import hashlib
from io import BytesIO

import streamlit as st

from data_diagnostics.analysis import profile_dataset
from data_diagnostics.ingestion import DatasetError, read_dataset


@st.cache_data(
    scope="session",
    show_spinner=False,
)
def load_and_profile_dataset(
    payload: bytes,
    filename: str,
):
    """Load and profile one uploaded dataset."""

    data = read_dataset(
        BytesIO(payload),
        filename=filename,
    )

    profile = profile_dataset(data)

    return data, profile


st.title("Data Diagnostics Platform")

st.write(
    "Upload a dataset and build a reproducible workspace for "
    "profiling, diagnostics, transformation, and analysis."
)

st.caption(
    "The raw uploaded dataset is preserved unchanged. "
    "Approved transformations create a separate working dataset."
)


uploaded_file = st.file_uploader(
    "Upload a dataset",
    type=["csv", "xlsx"],
    help="Currently supported formats: CSV and XLSX.",
)


if uploaded_file is not None:
    payload = uploaded_file.getvalue()

    fingerprint = hashlib.sha256(payload).hexdigest()

    dataset_changed = fingerprint != st.session_state.dataset_fingerprint

    if dataset_changed:
        try:
            with st.spinner("Loading and profiling dataset..."):
                data, profile = load_and_profile_dataset(
                    payload,
                    uploaded_file.name,
                )

        except DatasetError as exc:
            st.error(str(exc))
            st.stop()

        st.session_state.dataset_name = uploaded_file.name

        st.session_state.dataset_fingerprint = fingerprint

        st.session_state.raw_data = data.copy(deep=True)

        st.session_state.raw_profile = profile

        st.session_state.working_data = data.copy(deep=True)

        st.session_state.working_profile = profile

        st.session_state.transformation_plan = None
        st.session_state.transformation_result = None

        st.success(f"Loaded {uploaded_file.name}.")


raw_data = st.session_state.raw_data
raw_profile = st.session_state.raw_profile
working_data = st.session_state.working_data


if raw_data is None:
    st.info("Upload a CSV or XLSX file to create a workspace.")

    st.stop()


st.subheader("Workspace Overview")

metric_columns = st.columns(5)

metric_columns[0].metric(
    "Raw Rows",
    f"{len(raw_data):,}",
)

metric_columns[1].metric(
    "Raw Columns",
    f"{len(raw_data.columns):,}",
)

metric_columns[2].metric(
    "Missing Cells",
    f"{raw_profile.quality.missing_cell_count:,}",
)

metric_columns[3].metric(
    "Empty Rows",
    f"{raw_profile.quality.all_missing_row_count:,}",
)

metric_columns[4].metric(
    "Quality Findings",
    f"{raw_profile.quality.issue_count:,}",
)


st.subheader("Raw Data Preview")

st.dataframe(
    raw_data.head(20),
    width="stretch",
)


if working_data is not None:
    changed = not (raw_data.shape == working_data.shape and raw_data.equals(working_data))

    if changed:
        st.info(
            "A transformed working dataset is active. The raw uploaded dataset remains unchanged."
        )
    else:
        st.caption("The working dataset currently matches the raw dataset.")


st.subheader("Next Steps")

st.markdown(
    """
Use the navigation above to:

- **Profile** — inspect schema, quality findings, and descriptive statistics.
- **Transform** — review suggested cleaning actions and create a working dataset.
"""
)
