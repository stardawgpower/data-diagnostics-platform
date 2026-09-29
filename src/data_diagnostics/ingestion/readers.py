from pathlib import Path
from typing import IO

import pandas as pd

from data_diagnostics.ingestion.exceptions import (
    DatasetReadError,
    EmptyDatasetError,
    UnsupportedFileTypeError,
)

SUPPORTED_EXTENSIONS = {".csv", ".xlsx"}


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

    resolved_filename = _resolve_filename(source, filename)
    extension = Path(resolved_filename).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{extension}'. "
            f"Supported types are: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    try:
        if extension == ".csv":
            data = pd.read_csv(source)
        else:
            data = pd.read_excel(source, sheet_name=sheet_name)

    except (pd.errors.ParserError, UnicodeDecodeError, ValueError, OSError) as exc:
        raise DatasetReadError(f"Unable to read dataset '{resolved_filename}'.") from exc

    return _validate_dataframe(data)
