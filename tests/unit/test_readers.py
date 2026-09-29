from io import BytesIO, StringIO

import pandas as pd
import pytest

from data_diagnostics.ingestion import (
    DatasetReadError,
    EmptyDatasetError,
    UnsupportedFileTypeError,
    read_dataset,
)


def test_read_csv_from_file_like_object() -> None:
    source = StringIO("name,age,city\nAlice,30,Delhi\nBob,25,Mumbai\n")

    data = read_dataset(source, filename="people.csv")

    assert data.shape == (2, 3)
    assert data.columns.tolist() == ["name", "age", "city"]
    assert data["name"].tolist() == ["Alice", "Bob"]


def test_read_csv_from_path(tmp_path) -> None:
    file_path = tmp_path / "sample.csv"
    file_path.write_text(
        "product,price\nLaptop,80000\nPhone,40000\n",
        encoding="utf-8",
    )

    data = read_dataset(file_path)

    assert data.shape == (2, 2)
    assert data["product"].tolist() == ["Laptop", "Phone"]


def test_read_excel_file() -> None:
    source_data = pd.DataFrame(
        {
            "name": ["Alice", "Bob"],
            "score": [91, 87],
        }
    )

    buffer = BytesIO()

    source_data.to_excel(
        buffer,
        index=False,
        engine="openpyxl",
    )

    buffer.seek(0)

    loaded_data = read_dataset(
        buffer,
        filename="scores.xlsx",
    )

    pd.testing.assert_frame_equal(loaded_data, source_data)


def test_unsupported_file_type() -> None:
    source = StringIO("name,age\nAlice,30\n")

    with pytest.raises(UnsupportedFileTypeError):
        read_dataset(source, filename="people.json")


def test_empty_dataset() -> None:
    source = StringIO("name,age\n")

    with pytest.raises(EmptyDatasetError):
        read_dataset(source, filename="people.csv")


def test_filename_required_for_unnamed_file_object() -> None:
    source = StringIO("name,age\nAlice,30\n")

    with pytest.raises(DatasetReadError):
        read_dataset(source)
