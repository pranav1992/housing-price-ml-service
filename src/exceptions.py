"""Project-specific exceptions."""


class ConfigurationError(Exception):
    """Raised when application configuration is invalid."""


class DataIngestionError(Exception):
    """Raised when data ingestion cannot complete."""


class DataValidationError(Exception):
    """Raised when data validation fails."""


class DataTransformationError(Exception):
    """Raised when data transformation fails."""


class DataSplitError(Exception):
    """Raised when train/test splitting fails."""


class ModelTrainingError(Exception):
    """Raised when model training fails."""


class ModelEvaluationError(Exception):
    """Raised when model evaluation fails."""


class ModelInferenceError(Exception):
    """Raised when model inference fails."""
