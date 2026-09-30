class TransformationError(Exception):
    """Base exception for dataset transformation errors."""


class UnknownColumnError(TransformationError):
    """Raised when a transformation references a missing column."""


class InvalidTransformationError(TransformationError):
    """Raised when a requested transformation is invalid."""
