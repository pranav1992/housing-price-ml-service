"""Model inference services for the housing pipeline."""

from __future__ import annotations

import csv
import json
import pickle
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.configuration import DataInferenceConfig
from src.exceptions import ModelInferenceError


@dataclass(frozen=True, slots=True)
class ModelInferenceResult:
    """Outcome of a batch inference run."""

    predictions_path: Path
    metadata_path: Path
    model_name: str
    predicted_row_count: int


class ModelInferenceService:
    """Generate batch predictions from a trained model artifact."""

    def __init__(self, config: DataInferenceConfig) -> None:
        self._config = config

    def run(self) -> ModelInferenceResult:
        if not self._config.input_data_path.exists():
            raise ModelInferenceError(
                f"Inference input file not found: {self._config.input_data_path}"
            )
        if not self._config.model_artifact_path.exists():
            raise ModelInferenceError(
                f"Model artifact not found: {self._config.model_artifact_path}"
            )

        rows, fieldnames = self._read_rows(self._config.input_data_path)
        artifact = self._load_model_artifact(self._config.model_artifact_path)
        self._validate_artifact(artifact)

        target_column = artifact["target_column"]
        numeric_feature_columns = tuple(artifact["numeric_feature_columns"])
        categorical_feature_columns = tuple(artifact["categorical_feature_columns"])

        required_columns = {*numeric_feature_columns, *categorical_feature_columns}
        missing_columns = sorted(column for column in required_columns if column not in fieldnames)
        if missing_columns:
            raise ModelInferenceError(
                "Inference input is missing required columns: " + ", ".join(missing_columns)
            )

        feature_rows: list[dict[str, float | str]] = []
        for index, row in enumerate(rows, start=2):
            feature_rows.append(
                self._build_feature_row(
                    row,
                    numeric_feature_columns=numeric_feature_columns,
                    categorical_feature_columns=categorical_feature_columns,
                    index=index,
                )
            )

        pipeline = artifact["pipeline"]
        predictions = pipeline.predict(feature_rows)

        self._config.predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_predictions(
            rows=rows,
            predictions=predictions,
            target_column=target_column,
        )
        self._write_metadata(
            model_name=artifact["model_name"],
            target_column=target_column,
            numeric_feature_columns=numeric_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            predicted_row_count=len(rows),
            input_columns=fieldnames,
            target_present=target_column in fieldnames,
        )

        return ModelInferenceResult(
            predictions_path=self._config.predictions_path,
            metadata_path=self._config.metadata_path,
            model_name=artifact["model_name"],
            predicted_row_count=len(rows),
        )

    @staticmethod
    def _read_rows(input_path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
        with input_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = tuple(reader.fieldnames or ())
        if not rows:
            raise ModelInferenceError(f"Inference input file is empty: {input_path}")
        return rows, fieldnames

    @staticmethod
    def _load_model_artifact(model_artifact_path: Path) -> dict[str, Any]:
        try:
            with model_artifact_path.open("rb") as file_obj:
                return pickle.load(file_obj)
        except ModuleNotFoundError as exc:
            raise ModelInferenceError(
                "scikit-learn is required for model inference. Install project dependencies before running inference."
            ) from exc
        except Exception as exc:
            raise ModelInferenceError(
                f"Failed to load model artifact: {model_artifact_path}"
            ) from exc

    @staticmethod
    def _validate_artifact(artifact: dict[str, Any]) -> None:
        required_keys = {
            "model_name",
            "pipeline",
            "target_column",
            "numeric_feature_columns",
            "categorical_feature_columns",
        }
        missing = sorted(key for key in required_keys if key not in artifact)
        if missing:
            raise ModelInferenceError(
                "Model artifact is missing required keys: " + ", ".join(missing)
            )

    @staticmethod
    def _build_feature_row(
        row: dict[str, str],
        *,
        numeric_feature_columns: tuple[str, ...],
        categorical_feature_columns: tuple[str, ...],
        index: int,
    ) -> dict[str, float | str]:
        features: dict[str, float | str] = {}
        for column in numeric_feature_columns:
            raw_value = row.get(column, "").strip()
            try:
                features[column] = float(raw_value)
            except ValueError as exc:
                raise ModelInferenceError(
                    f"Numeric inference feature `{column}` contains a non-numeric value at line {index}: "
                    f"`{raw_value}`."
                ) from exc

        for column in categorical_feature_columns:
            features[column] = row.get(column, "").strip()

        return features

    def _write_predictions(
        self,
        *,
        rows: list[dict[str, str]],
        predictions,
        target_column: str,
    ) -> None:
        output_rows: list[dict[str, Any]] = []
        for row, predicted in zip(rows, predictions):
            output_row = dict(row)
            if target_column in output_row:
                output_row[f"actual_{target_column}"] = output_row.pop(target_column)
            output_row[f"predicted_{target_column}"] = float(predicted)
            output_rows.append(output_row)

        fieldnames = list(output_rows[0].keys()) if output_rows else [f"predicted_{target_column}"]
        with self._config.predictions_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(output_rows)

    def _write_metadata(
        self,
        *,
        model_name: str,
        target_column: str,
        numeric_feature_columns: tuple[str, ...],
        categorical_feature_columns: tuple[str, ...],
        predicted_row_count: int,
        input_columns: tuple[str, ...],
        target_present: bool,
    ) -> None:
        metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "input_data_path": str(self._config.input_data_path),
            "model_artifact_path": str(self._config.model_artifact_path),
            "predictions_path": str(self._config.predictions_path),
            "target_column": target_column,
            "numeric_feature_columns": list(numeric_feature_columns),
            "categorical_feature_columns": list(categorical_feature_columns),
            "input_columns": list(input_columns),
            "target_present": target_present,
            "predicted_row_count": predicted_row_count,
        }

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._config.metadata_path.parent,
            delete=False,
        ) as file_obj:
            json.dump(metadata, file_obj, indent=2)
            file_obj.write("\n")
            temp_name = file_obj.name

        Path(temp_name).replace(self._config.metadata_path)
