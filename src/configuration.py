"""Configuration loading for the data ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.exceptions import ConfigurationError


@dataclass(frozen=True, slots=True)
class DataIngestionConfig:
    """Configuration for downloading and materializing a dataset file."""

    source_name: str
    source_url: str
    raw_data_dir: Path
    extracted_data_dir: Path
    manifest_path: Path
    file_name: str

    @property
    def cache_file_path(self) -> Path:
        return self.raw_data_dir / self.file_name

    @property
    def downloaded_file_path(self) -> Path:
        return self.extracted_data_dir / self.file_name


@dataclass(frozen=True, slots=True)
class DataValidationConfig:
    """Configuration for validating the ingested dataset file."""

    data_file_path: Path
    required_columns: tuple[str, ...]
    non_empty_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...]
    datetime_columns: tuple[str, ...]
    strict_positive_columns: tuple[str, ...]
    warning_non_positive_columns: tuple[str, ...]
    expected_constant_values: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class DataTransformationConfig:
    """Configuration for transforming validated data into model-ready artifacts."""

    input_data_path: Path
    transformed_data_path: Path
    metadata_path: Path
    target_column: str
    date_column: str
    numeric_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    drop_columns: tuple[str, ...]
    statezip_column: str
    state_column_name: str
    zipcode_column_name: str
    drop_non_positive_target: bool


@dataclass(frozen=True, slots=True)
class DataSplitConfig:
    """Configuration for persisting a deterministic train/test split."""

    input_data_path: Path
    train_data_path: Path
    test_data_path: Path
    metadata_path: Path
    target_column: str
    test_size: float
    random_state: int


@dataclass(frozen=True, slots=True)
class DataTrainingConfig:
    """Configuration for training a regression model."""

    input_data_path: Path
    split_metadata_path: Path
    model_artifact_path: Path
    metadata_path: Path
    model_name: str
    target_column: str
    numeric_feature_columns: tuple[str, ...]
    categorical_feature_columns: tuple[str, ...]
    test_size: float
    random_state: int


@dataclass(frozen=True, slots=True)
class DataEvaluationConfig:
    """Configuration for evaluating a trained model on the held-out split."""

    input_data_path: Path
    split_metadata_path: Path
    model_artifact_path: Path
    metrics_path: Path
    metadata_path: Path
    sample_predictions_path: Path
    test_size: float
    random_state: int


@dataclass(frozen=True, slots=True)
class DataInferenceConfig:
    """Configuration for batch inference using a trained model artifact."""

    input_data_path: Path
    model_artifact_path: Path
    predictions_path: Path
    metadata_path: Path


def load_data_ingestion_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataIngestionConfig:
    """Load and validate ingestion settings from YAML."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_file}")

    root_dir = Path(project_root).expanduser().resolve() if project_root else config_file.parent.parent
    config_data = _read_yaml(config_file)

    ingestion_data = config_data.get("data_ingestion")
    if not isinstance(ingestion_data, dict):
        raise ConfigurationError("Missing `data_ingestion` section in config.")

    source_name = _require_string(ingestion_data, "source_name")
    source_url = _require_string(ingestion_data, "source_url")
    file_name = _require_string(ingestion_data, "file_name")
    raw_data_dir = _resolve_path(root_dir, _require_string(ingestion_data, "raw_data_dir"))
    extracted_data_dir = _resolve_path(root_dir, _require_string(ingestion_data, "extracted_data_dir"))
    manifest_path = _resolve_path(root_dir, _require_string(ingestion_data, "manifest_path"))

    return DataIngestionConfig(
        source_name=source_name,
        source_url=source_url,
        raw_data_dir=raw_data_dir,
        extracted_data_dir=extracted_data_dir,
        manifest_path=manifest_path,
        file_name=file_name,
    )


def load_data_validation_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataValidationConfig:
    """Load and validate data validation settings from YAML."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_file}")

    root_dir = Path(project_root).expanduser().resolve() if project_root else config_file.parent.parent
    config_data = _read_yaml(config_file)

    validation_data = config_data.get("data_validation")
    if not isinstance(validation_data, dict):
        raise ConfigurationError("Missing `data_validation` section in config.")

    return DataValidationConfig(
        data_file_path=_resolve_path(root_dir, _require_string(validation_data, "data_file_path")),
        required_columns=_require_csv_list(validation_data, "required_columns"),
        non_empty_columns=_require_csv_list(validation_data, "non_empty_columns"),
        numeric_columns=_require_csv_list(validation_data, "numeric_columns"),
        datetime_columns=_optional_csv_list(validation_data, "datetime_columns"),
        strict_positive_columns=_optional_csv_list(validation_data, "strict_positive_columns"),
        warning_non_positive_columns=_optional_csv_list(validation_data, "warning_non_positive_columns"),
        expected_constant_values=_optional_key_value_pairs(validation_data, "expected_constant_values"),
    )


def load_data_transformation_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataTransformationConfig:
    """Load and validate data transformation settings from YAML."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_file}")

    root_dir = Path(project_root).expanduser().resolve() if project_root else config_file.parent.parent
    config_data = _read_yaml(config_file)

    transformation_data = config_data.get("data_transformation")
    if not isinstance(transformation_data, dict):
        raise ConfigurationError("Missing `data_transformation` section in config.")

    return DataTransformationConfig(
        input_data_path=_resolve_path(root_dir, _require_string(transformation_data, "input_data_path")),
        transformed_data_path=_resolve_path(
            root_dir,
            _require_string(transformation_data, "transformed_data_path"),
        ),
        metadata_path=_resolve_path(root_dir, _require_string(transformation_data, "metadata_path")),
        target_column=_require_string(transformation_data, "target_column"),
        date_column=_require_string(transformation_data, "date_column"),
        numeric_columns=_require_csv_list_with_section(
            transformation_data,
            "numeric_columns",
            "data_transformation",
        ),
        categorical_columns=_require_csv_list_with_section(
            transformation_data,
            "categorical_columns",
            "data_transformation",
        ),
        drop_columns=_optional_csv_list_with_section(
            transformation_data,
            "drop_columns",
            "data_transformation",
        ),
        statezip_column=_require_string(transformation_data, "statezip_column"),
        state_column_name=_require_string(transformation_data, "state_column_name"),
        zipcode_column_name=_require_string(transformation_data, "zipcode_column_name"),
        drop_non_positive_target=_require_bool_with_section(
            transformation_data,
            "drop_non_positive_target",
            "data_transformation",
        ),
    )


def load_data_training_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataTrainingConfig:
    """Load and validate data training settings from YAML."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_file}")

    root_dir = Path(project_root).expanduser().resolve() if project_root else config_file.parent.parent
    config_data = _read_yaml(config_file)

    training_data = config_data.get("data_training")
    if not isinstance(training_data, dict):
        raise ConfigurationError("Missing `data_training` section in config.")

    return DataTrainingConfig(
        input_data_path=_resolve_path(
            root_dir,
            _require_string_with_section(training_data, "input_data_path", "data_training"),
        ),
        split_metadata_path=_resolve_path(
            root_dir,
            _require_string_with_section(training_data, "split_metadata_path", "data_training"),
        ),
        model_artifact_path=_resolve_path(
            root_dir,
            _require_string_with_section(training_data, "model_artifact_path", "data_training"),
        ),
        metadata_path=_resolve_path(
            root_dir,
            _require_string_with_section(training_data, "metadata_path", "data_training"),
        ),
        model_name=_require_string_with_section(training_data, "model_name", "data_training"),
        target_column=_require_string_with_section(training_data, "target_column", "data_training"),
        numeric_feature_columns=_require_csv_list_with_section(
            training_data,
            "numeric_feature_columns",
            "data_training",
        ),
        categorical_feature_columns=_optional_csv_list_with_section(
            training_data,
            "categorical_feature_columns",
            "data_training",
        ),
        test_size=_require_probability_with_section(training_data, "test_size", "data_training"),
        random_state=_require_int_with_section(training_data, "random_state", "data_training"),
    )


def load_data_evaluation_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataEvaluationConfig:
    """Load and validate model evaluation settings from YAML."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_file}")

    root_dir = Path(project_root).expanduser().resolve() if project_root else config_file.parent.parent
    config_data = _read_yaml(config_file)

    evaluation_data = config_data.get("data_evaluation")
    if not isinstance(evaluation_data, dict):
        raise ConfigurationError("Missing `data_evaluation` section in config.")

    return DataEvaluationConfig(
        input_data_path=_resolve_path(
            root_dir,
            _require_string_with_section(evaluation_data, "input_data_path", "data_evaluation"),
        ),
        split_metadata_path=_resolve_path(
            root_dir,
            _require_string_with_section(evaluation_data, "split_metadata_path", "data_evaluation"),
        ),
        model_artifact_path=_resolve_path(
            root_dir,
            _require_string_with_section(evaluation_data, "model_artifact_path", "data_evaluation"),
        ),
        metrics_path=_resolve_path(
            root_dir,
            _require_string_with_section(evaluation_data, "metrics_path", "data_evaluation"),
        ),
        metadata_path=_resolve_path(
            root_dir,
            _require_string_with_section(evaluation_data, "metadata_path", "data_evaluation"),
        ),
        sample_predictions_path=_resolve_path(
            root_dir,
            _require_string_with_section(evaluation_data, "sample_predictions_path", "data_evaluation"),
        ),
        test_size=_require_probability_with_section(evaluation_data, "test_size", "data_evaluation"),
        random_state=_require_int_with_section(evaluation_data, "random_state", "data_evaluation"),
    )


def load_data_inference_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataInferenceConfig:
    """Load and validate model inference settings from YAML."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_file}")

    root_dir = Path(project_root).expanduser().resolve() if project_root else config_file.parent.parent
    config_data = _read_yaml(config_file)

    inference_data = config_data.get("data_inference")
    if not isinstance(inference_data, dict):
        raise ConfigurationError("Missing `data_inference` section in config.")

    return DataInferenceConfig(
        input_data_path=_resolve_path(
            root_dir,
            _require_string_with_section(inference_data, "input_data_path", "data_inference"),
        ),
        model_artifact_path=_resolve_path(
            root_dir,
            _require_string_with_section(inference_data, "model_artifact_path", "data_inference"),
        ),
        predictions_path=_resolve_path(
            root_dir,
            _require_string_with_section(inference_data, "predictions_path", "data_inference"),
        ),
        metadata_path=_resolve_path(
            root_dir,
            _require_string_with_section(inference_data, "metadata_path", "data_inference"),
        ),
    )


def load_data_split_config(
    config_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> DataSplitConfig:
    """Load and validate data split settings from YAML."""

    config_file = Path(config_path).expanduser().resolve()
    if not config_file.exists():
        raise ConfigurationError(f"Config file not found: {config_file}")

    root_dir = Path(project_root).expanduser().resolve() if project_root else config_file.parent.parent
    config_data = _read_yaml(config_file)

    split_data = config_data.get("data_split")
    if not isinstance(split_data, dict):
        raise ConfigurationError("Missing `data_split` section in config.")

    return DataSplitConfig(
        input_data_path=_resolve_path(
            root_dir,
            _require_string_with_section(split_data, "input_data_path", "data_split"),
        ),
        train_data_path=_resolve_path(
            root_dir,
            _require_string_with_section(split_data, "train_data_path", "data_split"),
        ),
        test_data_path=_resolve_path(
            root_dir,
            _require_string_with_section(split_data, "test_data_path", "data_split"),
        ),
        metadata_path=_resolve_path(
            root_dir,
            _require_string_with_section(split_data, "metadata_path", "data_split"),
        ),
        target_column=_require_string_with_section(split_data, "target_column", "data_split"),
        test_size=_require_probability_with_section(split_data, "test_size", "data_split"),
        random_state=_require_int_with_section(split_data, "random_state", "data_split"),
    )


def _read_yaml(config_file: Path) -> dict[str, Any]:
    loaded: dict[str, Any] = {}
    current_section: dict[str, str] | None = None

    with config_file.open("r", encoding="utf-8") as file_obj:
        for line_number, raw_line in enumerate(file_obj, start=1):
            line = raw_line.rstrip()
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if not line.startswith(" "):
                if not stripped.endswith(":"):
                    raise ConfigurationError(
                        f"Invalid config format at line {line_number}: expected a top-level section."
                    )
                section_name = stripped[:-1].strip()
                current_section = {}
                loaded[section_name] = current_section
                continue

            if current_section is None:
                raise ConfigurationError(
                    f"Invalid config format at line {line_number}: key-value pair outside a section."
                )

            if ":" not in stripped:
                raise ConfigurationError(
                    f"Invalid config format at line {line_number}: expected `key: value`."
                )

            key, value = stripped.split(":", maxsplit=1)
            current_section[key.strip()] = value.strip()

    return loaded


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Missing or invalid `data_ingestion.{key}`.")
    return value.strip()


def _resolve_path(root_dir: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (root_dir / path).resolve()


def _require_csv_list(data: dict[str, Any], key: str) -> tuple[str, ...]:
    values = _optional_csv_list(data, key)
    if not values:
        raise ConfigurationError(f"Missing or invalid `data_validation.{key}`.")
    return values


def _optional_csv_list(data: dict[str, Any], key: str) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ConfigurationError(f"Missing or invalid `data_validation.{key}`.")

    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items


def _require_csv_list_with_section(
    data: dict[str, Any],
    key: str,
    section_name: str,
) -> tuple[str, ...]:
    values = _optional_csv_list_with_section(data, key, section_name)
    if not values:
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")
    return values


def _optional_csv_list_with_section(
    data: dict[str, Any],
    key: str,
    section_name: str,
) -> tuple[str, ...]:
    value = data.get(key)
    if value is None:
        return ()
    if not isinstance(value, str):
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _require_string_with_section(
    data: dict[str, Any],
    key: str,
    section_name: str,
) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")
    return value.strip()


def _require_float_with_section(
    data: dict[str, Any],
    key: str,
    section_name: str,
) -> float:
    value = data.get(key)
    if not isinstance(value, str):
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")
    try:
        return float(value.strip())
    except ValueError as exc:
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.") from exc


def _require_probability_with_section(
    data: dict[str, Any],
    key: str,
    section_name: str,
) -> float:
    value = _require_float_with_section(data, key, section_name)
    if not 0 < value < 1:
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")
    return value


def _require_int_with_section(
    data: dict[str, Any],
    key: str,
    section_name: str,
) -> int:
    value = data.get(key)
    if not isinstance(value, str):
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")
    try:
        return int(value.strip())
    except ValueError as exc:
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.") from exc


def _require_bool_with_section(
    data: dict[str, Any],
    key: str,
    section_name: str,
) -> bool:
    value = data.get(key)
    if not isinstance(value, str):
        raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")

    normalized = value.strip().lower()
    if normalized in {"true", "yes", "1"}:
        return True
    if normalized in {"false", "no", "0"}:
        return False

    raise ConfigurationError(f"Missing or invalid `{section_name}.{key}`.")


def _optional_key_value_pairs(data: dict[str, Any], key: str) -> tuple[tuple[str, str], ...]:
    pairs = []
    for item in _optional_csv_list(data, key):
        if "=" not in item:
            raise ConfigurationError(f"Invalid `data_validation.{key}` entry: `{item}`.")
        name, value = item.split("=", maxsplit=1)
        name = name.strip()
        value = value.strip()
        if not name or not value:
            raise ConfigurationError(f"Invalid `data_validation.{key}` entry: `{item}`.")
        pairs.append((name, value))
    return tuple(pairs)
