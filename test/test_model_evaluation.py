from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path

import pytest

from src.configuration import DataEvaluationConfig, load_data_evaluation_config
from src.exceptions import ModelEvaluationError
from src.model_evaluation import ModelEvaluationService


def test_evaluation_service_writes_metrics_and_predictions(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text(build_dataset(), encoding="utf-8")
    model_artifact_path = tmp_path / "artifacts" / "models" / "linear_regression_model.pkl"
    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    write_model_artifact(input_path=input_path, model_artifact_path=model_artifact_path)

    config = build_config(tmp_path, input_path=input_path, model_artifact_path=model_artifact_path)
    result = ModelEvaluationService(config).run()

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    with result.sample_predictions_path.open("r", encoding="utf-8", newline="") as file_obj:
        sample_predictions = list(csv.DictReader(file_obj))

    assert result.model_name == "linear_regression"
    assert result.evaluated_row_count == 2
    assert result.metrics_path.exists()
    assert result.metadata_path.exists()
    assert result.sample_predictions_path.exists()
    assert metrics["evaluated_row_count"] == 2
    assert "mae" in metrics and "rmse" in metrics and "r2" in metrics
    assert metadata["target_column"] == "price"
    assert len(sample_predictions) == 2
    assert "actual_price" in sample_predictions[0]
    assert "predicted_price" in sample_predictions[0]


def test_evaluation_service_fails_when_artifact_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text(build_dataset(), encoding="utf-8")
    config = build_config(
        tmp_path,
        input_path=input_path,
        model_artifact_path=tmp_path / "artifacts" / "models" / "missing.pkl",
    )

    with pytest.raises(ModelEvaluationError, match="Model artifact not found"):
        ModelEvaluationService(config).run()


def test_load_data_evaluation_config_resolves_paths_and_fields(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    config_dir.mkdir()
    config_file = config_dir / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_evaluation:",
                "  input_data_path: artifacts/processed/usa_housing_transformed.csv",
                "  model_artifact_path: artifacts/models/linear_regression_model.pkl",
                "  metrics_path: artifacts/evaluation/linear_regression_metrics.json",
                "  metadata_path: artifacts/evaluation/linear_regression_evaluation.metadata.json",
                "  sample_predictions_path: artifacts/evaluation/linear_regression_sample_predictions.csv",
                "  test_size: 0.2",
                "  random_state: 42",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_data_evaluation_config(config_file, project_root=project_root)

    assert loaded.input_data_path == project_root / "artifacts" / "processed" / "usa_housing_transformed.csv"
    assert loaded.metrics_path == project_root / "artifacts" / "evaluation" / "linear_regression_metrics.json"
    assert loaded.sample_predictions_path == project_root / "artifacts" / "evaluation" / "linear_regression_sample_predictions.csv"
    assert loaded.test_size == 0.2
    assert loaded.random_state == 42


def build_config(base_dir: Path, *, input_path: Path, model_artifact_path: Path) -> DataEvaluationConfig:
    return DataEvaluationConfig(
        input_data_path=input_path,
        model_artifact_path=model_artifact_path,
        metrics_path=base_dir / "artifacts" / "evaluation" / "linear_regression_metrics.json",
        metadata_path=base_dir / "artifacts" / "evaluation" / "linear_regression_evaluation.metadata.json",
        sample_predictions_path=base_dir / "artifacts" / "evaluation" / "linear_regression_sample_predictions.csv",
        test_size=0.25,
        random_state=7,
    )


def write_model_artifact(*, input_path: Path, model_artifact_path: Path) -> None:
    from src.model_training import ModelTrainingService

    training_config = build_training_config(model_artifact_path.parent.parent.parent, input_path)
    result = ModelTrainingService(training_config).run()
    payload = pickle.loads(result.model_artifact_path.read_bytes())
    model_artifact_path.write_bytes(pickle.dumps(payload))


def build_dataset() -> str:
    return "\n".join(
        [
            "bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,city,state,zipcode,sale_year,sale_month,sale_day,price",
            "3,2,1340,1384,3,0,0,3,1340,0,2008,0,Seattle,WA,98103,2014,5,9,376000",
            "4,3.25,3540,159430,2,0,0,3,3540,0,2007,0,Carnation,WA,98014,2014,5,9,800000",
            "5,6.5,7270,130017,2,0,0,3,6420,850,2010,0,Issaquah,WA,98029,2014,5,9,2238888",
            "3,2.5,3370,7911,1,0,0,3,1670,1700,1968,0,Bellevue,WA,98008,2014,5,9,670000",
            "3,2.25,1710,6622,1,0,0,4,1710,0,1976,0,Redmond,WA,98052,2014,5,9,530000",
            "2,1,900,5000,1,0,0,3,900,0,1940,0,Seattle,WA,98115,2014,5,10,250000",
            "4,2.5,2500,7000,2,0,0,3,2500,0,1995,0,Renton,WA,98058,2014,5,10,480000",
            "3,1.75,1600,6200,1,0,0,4,1100,500,1955,0,Kent,WA,98031,2014,5,11,315000",
            "",
        ]
    )


def build_training_config(base_dir: Path, input_path: Path) -> DataTrainingConfig:
    from src.configuration import DataTrainingConfig

    return DataTrainingConfig(
        input_data_path=input_path,
        model_artifact_path=base_dir / "artifacts" / "models" / "linear_regression_model.pkl",
        metadata_path=base_dir / "artifacts" / "models" / "linear_regression_model.metadata.json",
        model_name="linear_regression",
        target_column="price",
        numeric_feature_columns=(
            "bedrooms",
            "bathrooms",
            "sqft_living",
            "sqft_lot",
            "floors",
            "waterfront",
            "view",
            "condition",
            "sqft_above",
            "sqft_basement",
            "yr_built",
            "yr_renovated",
            "sale_year",
            "sale_month",
            "sale_day",
        ),
        categorical_feature_columns=("city", "state"),
        test_size=0.25,
        random_state=7,
    )
