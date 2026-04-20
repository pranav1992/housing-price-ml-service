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
