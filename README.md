# Data Diagnostics Platform

A modular data diagnostics and analytics platform for inspecting tabular datasets,
identifying data-quality problems, applying explicit transformations, and preparing
clean working data for exploratory analysis and machine-learning workflows.

The project is a ground-up redesign of an analytics application originally developed
during a data science internship in 2023.

## Current Status

Active development — 2026.

The current platform supports:

- CSV and XLSX ingestion
- delimiter and encoding handling for CSV files
- schema and semantic-type inference
- data-quality diagnostics
- dataset and column profiling
- possible sentinel-value detection
- explicit sentinel-to-missing transformations
- removal of completely empty rows
- explicit column removal
- immutable raw-data preservation
- reproducible transformation plans
- transformation audit logs
- separate raw and working datasets
- multipage Streamlit interface

Planned modules include exploratory analysis, statistical relationship analysis,
temporal analysis, machine-learning workflows, association-rule mining, scenario
analysis, and automated reporting.

## Design Principle

The platform distinguishes between observation and transformation.

Dataset ingestion, schema inference, profiling, and quality diagnostics do not silently
alter uploaded data.

The application maintains two dataset states:

```text
Uploaded Dataset
      |
      v
Immutable Raw Dataset
      |
      +--> Schema Inference
      |
      +--> Quality Diagnostics
      |
      +--> Dataset Profiling
      |
      v
Explicit Transformation Plan
      |
      v
Working Dataset
      |
      +--> Exploratory Analysis
      +--> Statistical Analysis
      +--> Machine Learning
      +--> Reporting
```

The working dataset is regenerated from the raw dataset and the selected transformation
plan rather than by incrementally mutating previously cleaned data.

## Transformation Workflow

Current transformation operations include:

```text
Possible sentinel values
        |
        | user reviews and explicitly selects
        v
Replace selected values with missing data
        |
        v
Remove completely empty rows
        |
        v
Drop explicitly selected columns
        |
        v
Working Dataset
```

Possible sentinel values are diagnostic findings only. They are not replaced
automatically.

## Project Structure

```text
data-diagnostics-platform/
├── app/
│   ├── streamlit_app.py
│   └── views/
│       ├── workspace.py
│       ├── profile.py
│       └── transform.py
├── src/
│   └── data_diagnostics/
│       ├── analysis/
│       ├── association/
│       ├── ingestion/
│       ├── modeling/
│       ├── quality/
│       ├── reporting/
│       └── transformation/
├── tests/
│   ├── integration/
│   └── unit/
├── docs/
├── examples/
├── pyproject.toml
└── README.md
```

## Requirements

Python 3.12 is required.

The project currently targets:

```text
Python >=3.12,<3.13
```

## Local Development

Clone the repository and enter the project directory:

```bash
git clone https://github.com/stardawgpower/data-diagnostics-platform.git
cd data-diagnostics-platform
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Upgrade pip and install the project with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run the Application

Start the Streamlit application with:

```bash
streamlit run app/streamlit_app.py
```

Then upload a CSV or XLSX dataset through the Workspace page.

## Development Checks

Format the code:

```bash
ruff format .
```

Check linting:

```bash
ruff check .
```

Run the test suite:

```bash
pytest -q
```

Run the same checks used by CI:

```bash
ruff format --check .
ruff check .
pytest -q
```

## Continuous Integration

GitHub Actions runs formatting validation, Ruff linting, and the pytest suite for pull
requests targeting `main` and for pushes to `main`.

Workflow:

```text
Install Python 3.12
        |
        v
Install project + dev dependencies
        |
        v
Ruff formatting check
        |
        v
Ruff lint check
        |
        v
pytest
```

## Current Application Pages

### Workspace

Uploads and initializes a dataset workspace while preserving an immutable raw copy.

### Profile

Displays inferred schema, quality findings, descriptive summaries, and allows inspection
of either the raw or transformed working dataset.

### Transform

Builds and applies an explicit transformation plan. Supported operations currently
include sentinel replacement, completely empty-row removal, and column removal.

## Testing

The test suite covers ingestion, schema inference, quality diagnostics, profiling, and
transformation behavior.

Transformation tests include:

- raw-data immutability
- ordered transformation execution
- sentinel replacement
- selected-column isolation
- completely empty-row removal
- column removal
- unknown-column validation
- duplicate-column validation
- invalid replacement-value validation

## Roadmap

Upcoming development will focus on:

1. exploratory data analysis
2. relationship and correlation diagnostics
3. temporal analysis
4. regression and classification workflows
5. clustering
6. association-rule mining
7. scenario analysis
8. automated reporting
9. deployment and portfolio documentation

## License

A license has not yet been selected.