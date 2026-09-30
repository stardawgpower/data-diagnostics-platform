import csv
import re
from io import StringIO
from pathlib import Path
from typing import IO

import pandas as pd

from data_diagnostics.ingestion.exceptions import (
    DatasetReadError,
    EmptyDatasetError,
    UnsupportedFileTypeError,
)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}

_CSV_DELIMITERS = (",", ";", "\t", "|")

_DECIMAL_COMMA_PATTERN = re.compile(r"^[+-]?\d+,\d+(?:[eE][+-]?\d+)?$")

_DECIMAL_DOT_PATTERN = re.compile(r"^[+-]?\d+\.\d+(?:[eE][+-]?\d+)?$")


def _resolve_filename(
    source: str | Path | IO[bytes] | IO[str],
    filename: str | None,
) -> str:
    """Determine the filename used to identify the dataset format."""

    if filename:
        return filename

    if isinstance(source, (str, Path)):
        return str(source)

    source_name = getattr(source, "name", None)

    if source_name:
        return str(source_name)

    raise DatasetReadError("A filename is required when the dataset source does not provide one.")


def _decode_sample(
    sample: bytes | str,
) -> tuple[str, str | None]:
    """Decode a CSV sample and return its text and detected encoding."""

    if isinstance(sample, str):
        return sample, None

    try:
        return sample.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        return sample.decode("latin-1"), "latin-1"


def _read_text_sample(
    source: str | Path | IO[bytes] | IO[str],
    *,
    sample_size: int = 65_536,
) -> tuple[str, str | None]:
    """Read a small portion of a CSV without consuming the source."""

    if isinstance(source, (str, Path)):
        with Path(source).open("rb") as file:
            sample = file.read(sample_size)

        return _decode_sample(sample)

    try:
        source.seek(0)
        sample = source.read(sample_size)
        source.seek(0)
    except (AttributeError, OSError) as exc:
        raise DatasetReadError("The uploaded dataset must support reading and seeking.") from exc

    return _decode_sample(sample)


def _detect_csv_delimiter(sample: str) -> str:
    """Infer a CSV field delimiter from a text sample."""

    try:
        dialect = csv.Sniffer().sniff(
            sample,
            delimiters="".join(_CSV_DELIMITERS),
        )

        return dialect.delimiter

    except csv.Error:
        first_non_empty_line = next(
            (line for line in sample.splitlines() if line.strip()),
            "",
        )

        delimiter_counts = {
            delimiter: first_non_empty_line.count(delimiter) for delimiter in _CSV_DELIMITERS
        }

        best_delimiter = max(
            delimiter_counts,
            key=delimiter_counts.get,
        )

        if delimiter_counts[best_delimiter] == 0:
            return ","

        return best_delimiter


def _detect_decimal_separator(
    sample: str,
    delimiter: str,
) -> str:
    """Infer whether numeric fields use dot or comma decimals."""

    comma_decimal_count = 0
    dot_decimal_count = 0

    try:
        rows = csv.reader(
            StringIO(sample),
            delimiter=delimiter,
        )

        for row_index, row in enumerate(rows):
            if row_index == 0:
                continue

            for value in row:
                token = value.strip()

                if _DECIMAL_COMMA_PATTERN.fullmatch(token):
                    comma_decimal_count += 1

                elif _DECIMAL_DOT_PATTERN.fullmatch(token):
                    dot_decimal_count += 1

    except csv.Error:
        return "."

    if comma_decimal_count > 0 and comma_decimal_count > dot_decimal_count:
        return ","

    return "."


def _rewind_source(
    source: str | Path | IO[bytes] | IO[str],
) -> None:
    """Rewind a file-like object before parsing."""

    if isinstance(source, (str, Path)):
        return

    try:
        source.seek(0)
    except (AttributeError, OSError) as exc:
        raise DatasetReadError("The uploaded dataset could not be rewound before reading.") from exc


def _read_csv(
    source: str | Path | IO[bytes] | IO[str],
) -> pd.DataFrame:
    """Read a CSV using automatically detected dialect information."""

    sample, encoding = _read_text_sample(source)

    delimiter = _detect_csv_delimiter(sample)

    decimal_separator = _detect_decimal_separator(
        sample,
        delimiter,
    )

    _rewind_source(source)

    return pd.read_csv(
        source,
        sep=delimiter,
        decimal=decimal_separator,
        encoding=encoding,
    )


def _validate_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure the loaded dataset contains both rows and columns."""

    if data.shape[1] == 0:
        raise EmptyDatasetError("The dataset contains no columns.")

    if data.shape[0] == 0:
        raise EmptyDatasetError("The dataset contains no data rows.")

    return data


def read_dataset(
    source: str | Path | IO[bytes] | IO[str],
    filename: str | None = None,
    *,
    sheet_name: int | str = 0,
) -> pd.DataFrame:
    """
    Read a CSV or Excel dataset into a pandas DataFrame.

    CSV delimiter and decimal conventions are detected automatically.

    Parameters
    ----------
    source:
        File path or file-like object containing the dataset.

    filename:
        Optional filename used to determine the file extension.
        This is useful for uploaded file-like objects.

    sheet_name:
        Excel worksheet name or index. Defaults to the first worksheet.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.

    Raises
    ------
    UnsupportedFileTypeError
        If the file extension is unsupported.

    DatasetReadError
        If the dataset cannot be parsed.

    EmptyDatasetError
        If the dataset contains no rows or columns.
    """

    resolved_filename = _resolve_filename(
        source,
        filename,
    )

    extension = Path(resolved_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. "
            f"Supported types are: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    try:
        if extension == ".csv":
            data = _read_csv(source)

        else:
            _rewind_source(source)

            data = pd.read_excel(
                source,
                sheet_name=sheet_name,
            )

    except (
        pd.errors.ParserError,
        UnicodeDecodeError,
        ValueError,
        OSError,
    ) as exc:
        raise DatasetReadError(f"Unable to read dataset '{resolved_filename}'.") from exc

    return _validate_dataframe(data)
