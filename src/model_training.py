"""Model training services for the housing pipeline."""

from __future__ import annotations

import csv
import json
import pickle
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.configuration import DataTrainingConfig
from src.exceptions import ModelTrainingError


@dataclass(frozen=True, slots=True)
class ModelTrainingResult:
    """Outcome of a training run."""

    model_artifact_path: Path
    metadata_path: Path
    model_name: str
    input_row_count: int
    train_row_count: int
    test_row_count: int
    feature_count: int
    feature_names: tuple[str, ...]


class ModelTrainingService:
    """Train a deterministic regression model from transformed features."""

    def __init__(self, config: DataTrainingConfig) -> None:
        self._config = config

    def run(self) -> ModelTrainingResult:
        input_path = self._config.input_data_path
        if not input_path.exists():
            raise ModelTrainingError(f"Training input file not found: {input_path}")

        rows, fieldnames = self._read_rows(input_path)
        self._validate_required_columns(fieldnames)

        feature_rows: list[dict[str, float | str]] = []
        targets: list[float] = []
        for index, row in enumerate(rows, start=2):
            feature_rows.append(self._build_feature_row(row, index=index))
            targets.append(self._parse_target(row, index=index))

        if len(feature_rows) < 2:
            raise ModelTrainingError("Training requires at least two rows after transformation.")

        split_metadata = self._load_split_metadata()
        train_row_count = split_metadata.get("train_row_count", len(feature_rows))
        test_row_count = split_metadata.get("test_row_count", 0)
        if train_row_count != len(feature_rows):
            raise ModelTrainingError(
                "Training input row count does not match split metadata train_row_count."
            )

        model = self._build_pipeline()
        model.fit(feature_rows, targets)

        feature_names = tuple(model.named_steps["vectorizer"].get_feature_names_out())
        self._config.model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_model_artifact(model)
        self._write_metadata(
            input_row_count=len(feature_rows),
            train_row_count=train_row_count,
            test_row_count=test_row_count,
            feature_names=feature_names,
        )

        return ModelTrainingResult(
            model_artifact_path=self._config.model_artifact_path,
            metadata_path=self._config.metadata_path,
            model_name=self._config.model_name,
            input_row_count=len(feature_rows),
            train_row_count=train_row_count,
            test_row_count=test_row_count,
            feature_count=len(feature_names),
            feature_names=feature_names,
        )

    def _build_pipeline(self):
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
            from sklearn.feature_extraction import DictVectorizer
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler
        except ImportError as exc:
            raise ModelTrainingError(
                "scikit-learn is required for model training. Install project dependencies before running training."
            ) from exc

        if self._config.model_name == "linear_regression":
            return Pipeline(
                steps=[
                    ("vectorizer", DictVectorizer(sparse=True)),
                    ("scaler", StandardScaler(with_mean=False)),
                    ("regressor", LinearRegression()),
                ]
            )

        if self._config.model_name == "random_forest":
            return Pipeline(
                steps=[
                    ("vectorizer", DictVectorizer(sparse=True)),
                    (
                        "regressor",
                        RandomForestRegressor(
                            n_estimators=200,
                            random_state=self._config.random_state,
                            n_jobs=1,
                        ),
                    ),
                ]
            )

        if self._config.model_name == "hist_gradient_boosting":
            return Pipeline(
                steps=[
                    ("vectorizer", DictVectorizer(sparse=False)),
                    (
                        "regressor",
                        HistGradientBoostingRegressor(
                            random_state=self._config.random_state,
                        ),
                    ),
                ]
            )

        raise ModelTrainingError(
            f"Unsupported model `{self._config.model_name}`. "
            "Supported models: linear_regression, random_forest, hist_gradient_boosting."
        )

    @staticmethod
    def _read_rows(input_path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
        with input_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = tuple(reader.fieldnames or ())
        if not rows:
            raise ModelTrainingError(f"Training input file is empty: {input_path}")
        return rows, fieldnames

    def _validate_required_columns(self, fieldnames: tuple[str, ...]) -> None:
        required = {
            self._config.target_column,
            *self._config.numeric_feature_columns,
            *self._config.categorical_feature_columns,
        }
        missing = sorted(column for column in required if column not in fieldnames)
        if missing:
            raise ModelTrainingError(
                "Training input is missing required columns: " + ", ".join(missing)
            )

    def _build_feature_row(self, row: dict[str, str], *, index: int) -> dict[str, float | str]:
        features: dict[str, float | str] = {}

        for column in self._config.numeric_feature_columns:
            raw_value = row.get(column, "").strip()
            try:
                features[column] = float(raw_value)
            except ValueError as exc:
                raise ModelTrainingError(
                    f"Numeric training feature `{column}` contains a non-numeric value at line {index}: "
                    f"`{raw_value}`."
                ) from exc

        for column in self._config.categorical_feature_columns:
            features[column] = row.get(column, "").strip()

        return features

    def _parse_target(self, row: dict[str, str], *, index: int) -> float:
        raw_value = row.get(self._config.target_column, "").strip()
        try:
            return float(raw_value)
        except ValueError as exc:
            raise ModelTrainingError(
                f"Target column `{self._config.target_column}` contains a non-numeric value at line {index}: "
                f"`{raw_value}`."
            ) from exc

    def _load_split_metadata(self) -> dict[str, Any]:
        split_metadata_path = self._config.split_metadata_path
        if not split_metadata_path.exists():
            raise ModelTrainingError(f"Split metadata file not found: {split_metadata_path}")

        try:
            return json.loads(split_metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelTrainingError(
                f"Split metadata file is invalid JSON: {split_metadata_path}"
            ) from exc

    def _write_model_artifact(self, model) -> None:
        artifact_payload = {
            "model_name": self._config.model_name,
            "pipeline": model,
            "target_column": self._config.target_column,
            "numeric_feature_columns": self._config.numeric_feature_columns,
            "categorical_feature_columns": self._config.categorical_feature_columns,
        }

        with self._config.model_artifact_path.open("wb") as file_obj:
            pickle.dump(artifact_payload, file_obj)

    def _write_metadata(
        self,
        *,
        input_row_count: int,
        train_row_count: int,
        test_row_count: int,
        feature_names: tuple[str, ...],
    ) -> None:
        metadata: dict[str, Any] = {
            "generated_at": datetime.now(UTC).isoformat(),
            "model_name": self._config.model_name,
            "input_data_path": str(self._config.input_data_path),
            "model_artifact_path": str(self._config.model_artifact_path),
            "input_row_count": input_row_count,
            "train_row_count": train_row_count,
            "test_row_count": test_row_count,
            "target_column": self._config.target_column,
            "numeric_feature_columns": list(self._config.numeric_feature_columns),
            "categorical_feature_columns": list(self._config.categorical_feature_columns),
            "encoded_feature_count": len(feature_names),
            "encoded_feature_names": list(feature_names),
            "test_size": self._config.test_size,
            "random_state": self._config.random_state,
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
