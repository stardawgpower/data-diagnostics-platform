class DatasetError(Exception):
    """Base exception for dataset ingestion errors."""


class UnsupportedFileTypeError(DatasetError):
    """Raised when the uploaded dataset format is not supported."""


class DatasetReadError(DatasetError):
    """Raised when a supported dataset cannot be read."""


class EmptyDatasetError(DatasetError):
    """Raised when a dataset contains no usable rows or columns."""
