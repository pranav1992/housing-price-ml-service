"""Project-specific exceptions."""


class ConfigurationError(Exception):
    """Raised when application configuration is invalid."""


class DataIngestionError(Exception):
    """Raised when data ingestion cannot complete."""


class DataValidationError(Exception):
    """Raised when data validation fails."""
