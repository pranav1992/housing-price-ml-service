"""Model evaluation services for the housing pipeline."""

from __future__ import annotations

import csv
import json
import pickle
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.configuration import DataEvaluationConfig
from src.exceptions import ModelEvaluationError


@dataclass(frozen=True, slots=True)
class ModelEvaluationResult:
    """Outcome of a model evaluation run."""

    metrics_path: Path
    metadata_path: Path
    sample_predictions_path: Path
    model_name: str
    evaluated_row_count: int
    mae: float
    rmse: float
    r2: float


class ModelEvaluationService:
    """Evaluate a trained model on the deterministic held-out split."""

    def __init__(self, config: DataEvaluationConfig) -> None:
        self._config = config

    def run(self) -> ModelEvaluationResult:
        if not self._config.input_data_path.exists():
            raise ModelEvaluationError(
                f"Evaluation input file not found: {self._config.input_data_path}"
            )
        if not self._config.model_artifact_path.exists():
            raise ModelEvaluationError(
                f"Model artifact not found: {self._config.model_artifact_path}"
            )

        rows, fieldnames = self._read_rows(self._config.input_data_path)
        artifact = self._load_model_artifact(self._config.model_artifact_path)
        self._validate_artifact(artifact)

        target_column = artifact["target_column"]
        numeric_feature_columns = tuple(artifact["numeric_feature_columns"])
        categorical_feature_columns = tuple(artifact["categorical_feature_columns"])

        required_columns = {target_column, *numeric_feature_columns, *categorical_feature_columns}
        missing_columns = sorted(column for column in required_columns if column not in fieldnames)
        if missing_columns:
            raise ModelEvaluationError(
                "Evaluation input is missing required columns: " + ", ".join(missing_columns)
            )

        feature_rows: list[dict[str, float | str]] = []
        targets: list[float] = []
        for index, row in enumerate(rows, start=2):
            feature_rows.append(
                self._build_feature_row(
                    row,
                    numeric_feature_columns=numeric_feature_columns,
                    categorical_feature_columns=categorical_feature_columns,
                    index=index,
                )
            )
            targets.append(self._parse_target(row, target_column=target_column, index=index))

        train_test_split = self._load_train_test_split()
        _, X_test, _, y_test = train_test_split(
            feature_rows,
            targets,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
        )

        pipeline = artifact["pipeline"]
        predictions = pipeline.predict(X_test)
        mae, rmse, r2 = self._compute_metrics(y_test, predictions)

        self._config.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.sample_predictions_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_metrics(
            model_name=artifact["model_name"],
            evaluated_row_count=len(y_test),
            mae=mae,
            rmse=rmse,
            r2=r2,
        )
        self._write_metadata(
            model_name=artifact["model_name"],
            target_column=target_column,
            numeric_feature_columns=numeric_feature_columns,
            categorical_feature_columns=categorical_feature_columns,
            evaluated_row_count=len(y_test),
        )
        self._write_sample_predictions(X_test, y_test, predictions)

        return ModelEvaluationResult(
            metrics_path=self._config.metrics_path,
            metadata_path=self._config.metadata_path,
            sample_predictions_path=self._config.sample_predictions_path,
            model_name=artifact["model_name"],
            evaluated_row_count=len(y_test),
            mae=mae,
            rmse=rmse,
            r2=r2,
        )

    @staticmethod
    def _read_rows(input_path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
        with input_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = tuple(reader.fieldnames or ())
        if not rows:
            raise ModelEvaluationError(f"Evaluation input file is empty: {input_path}")
        return rows, fieldnames

    @staticmethod
    def _load_model_artifact(model_artifact_path: Path) -> dict[str, Any]:
        try:
            with model_artifact_path.open("rb") as file_obj:
                return pickle.load(file_obj)
        except ModuleNotFoundError as exc:
            raise ModelEvaluationError(
                "scikit-learn is required for model evaluation. Install project dependencies before running evaluation."
            ) from exc
        except Exception as exc:
            raise ModelEvaluationError(
                f"Failed to load model artifact: {model_artifact_path}"
            ) from exc

    @staticmethod
    def _validate_artifact(artifact: dict[str, Any]) -> None:
        required_keys = {"model_name", "pipeline", "target_column", "numeric_feature_columns", "categorical_feature_columns"}
        missing = sorted(key for key in required_keys if key not in artifact)
        if missing:
            raise ModelEvaluationError(
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
                raise ModelEvaluationError(
                    f"Numeric evaluation feature `{column}` contains a non-numeric value at line {index}: "
                    f"`{raw_value}`."
                ) from exc

        for column in categorical_feature_columns:
            features[column] = row.get(column, "").strip()

        return features

    @staticmethod
    def _parse_target(row: dict[str, str], *, target_column: str, index: int) -> float:
        raw_value = row.get(target_column, "").strip()
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ModelEvaluationError(
                f"Evaluation target `{target_column}` contains a non-numeric value at line {index}: "
                f"`{raw_value}`."
            ) from exc

    @staticmethod
    def _load_train_test_split():
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise ModelEvaluationError(
                "scikit-learn is required for model evaluation. Install project dependencies before running evaluation."
            ) from exc
        return train_test_split

    @staticmethod
    def _compute_metrics(y_true: list[float], predictions) -> tuple[float, float, float]:
        try:
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        except ImportError as exc:
            raise ModelEvaluationError(
                "scikit-learn is required for model evaluation. Install project dependencies before running evaluation."
            ) from exc

        mae = float(mean_absolute_error(y_true, predictions))
        rmse = float(mean_squared_error(y_true, predictions) ** 0.5)
        r2 = float(r2_score(y_true, predictions))
        return mae, rmse, r2

    def _write_metrics(
        self,
        *,
        model_name: str,
        evaluated_row_count: int,
        mae: float,
        rmse: float,
        r2: float,
    ) -> None:
        metrics = {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "evaluated_row_count": evaluated_row_count,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._config.metrics_path.parent,
            delete=False,
        ) as file_obj:
            json.dump(metrics, file_obj, indent=2)
            file_obj.write("\n")
            temp_name = file_obj.name

        Path(temp_name).replace(self._config.metrics_path)

    def _write_metadata(
        self,
        *,
        model_name: str,
        target_column: str,
        numeric_feature_columns: tuple[str, ...],
        categorical_feature_columns: tuple[str, ...],
        evaluated_row_count: int,
    ) -> None:
        metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_name": model_name,
            "input_data_path": str(self._config.input_data_path),
            "model_artifact_path": str(self._config.model_artifact_path),
            "target_column": target_column,
            "numeric_feature_columns": list(numeric_feature_columns),
            "categorical_feature_columns": list(categorical_feature_columns),
            "test_size": self._config.test_size,
            "random_state": self._config.random_state,
            "evaluated_row_count": evaluated_row_count,
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

    def _write_sample_predictions(
        self,
        X_test: list[dict[str, float | str]],
        y_test: list[float],
        predictions,
    ) -> None:
        rows = []
        for features, actual, predicted in zip(X_test[:10], y_test[:10], predictions[:10]):
            row = {
                "actual_price": actual,
                "predicted_price": float(predicted),
                **features,
            }
            rows.append(row)

        fieldnames = list(rows[0].keys()) if rows else ["actual_price", "predicted_price"]
        with self._config.sample_predictions_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
