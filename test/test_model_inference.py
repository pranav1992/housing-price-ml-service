from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.configuration import DataInferenceConfig, load_data_inference_config
from src.exceptions import ModelInferenceError
from src.model_inference import ModelInferenceService


def test_inference_service_writes_predictions_and_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "inference.csv"
    input_path.write_text(build_inference_dataset(), encoding="utf-8")
    model_artifact_path = tmp_path / "artifacts" / "models" / "linear_regression_model.pkl"
    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    write_model_artifact(model_artifact_path=model_artifact_path)

    config = build_config(tmp_path, input_path=input_path, model_artifact_path=model_artifact_path)
    result = ModelInferenceService(config).run()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    with result.predictions_path.open("r", encoding="utf-8", newline="") as file_obj:
        prediction_rows = list(csv.DictReader(file_obj))

    assert result.model_name == "linear_regression"
    assert result.predicted_row_count == 2
    assert metadata["predicted_row_count"] == 2
    assert metadata["target_present"] is False
    assert "predicted_price" in prediction_rows[0]
    assert "actual_price" not in prediction_rows[0]


def test_inference_service_fails_when_required_feature_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "inference.csv"
    input_path.write_text("bedrooms,city,state\n3,Seattle,WA\n", encoding="utf-8")
    model_artifact_path = tmp_path / "artifacts" / "models" / "linear_regression_model.pkl"
    model_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    write_model_artifact(model_artifact_path=model_artifact_path)

    config = build_config(tmp_path, input_path=input_path, model_artifact_path=model_artifact_path)

    with pytest.raises(ModelInferenceError, match="missing required columns"):
        ModelInferenceService(config).run()


def test_load_data_inference_config_resolves_paths_and_fields(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    config_dir.mkdir()
    config_file = config_dir / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_inference:",
                "  input_data_path: artifacts/splits/usa_housing_test.csv",
                "  model_artifact_path: artifacts/models/linear_regression_model.pkl",
                "  predictions_path: artifacts/predictions/linear_regression_predictions.csv",
                "  metadata_path: artifacts/predictions/linear_regression_predictions.metadata.json",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_data_inference_config(config_file, project_root=project_root)

    assert loaded.input_data_path == project_root / "artifacts" / "splits" / "usa_housing_test.csv"
    assert loaded.model_artifact_path == project_root / "artifacts" / "models" / "linear_regression_model.pkl"
    assert loaded.predictions_path == project_root / "artifacts" / "predictions" / "linear_regression_predictions.csv"
    assert loaded.metadata_path == project_root / "artifacts" / "predictions" / "linear_regression_predictions.metadata.json"


def build_config(base_dir: Path, *, input_path: Path, model_artifact_path: Path) -> DataInferenceConfig:
    return DataInferenceConfig(
        input_data_path=input_path,
        model_artifact_path=model_artifact_path,
        predictions_path=base_dir / "artifacts" / "predictions" / "linear_regression_predictions.csv",
        metadata_path=base_dir / "artifacts" / "predictions" / "linear_regression_predictions.metadata.json",
    )


def write_model_artifact(*, model_artifact_path: Path) -> None:
    from src.configuration import DataTrainingConfig
    from src.model_training import ModelTrainingService

    training_input_path = model_artifact_path.parent.parent.parent / "train.csv"
    training_input_path.write_text(build_training_dataset(), encoding="utf-8")
    split_metadata_path = (
        model_artifact_path.parent.parent.parent / "artifacts" / "splits" / "usa_housing_split.metadata.json"
    )
    split_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    split_metadata_path.write_text(
        json.dumps(
            {
                "train_row_count": 6,
                "test_row_count": 2,
                "test_size": 0.25,
                "random_state": 7,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    training_config = DataTrainingConfig(
        input_data_path=training_input_path,
        split_metadata_path=split_metadata_path,
        model_artifact_path=model_artifact_path.parent.parent.parent
        / "artifacts"
        / "models"
        / "linear_regression_model.pkl",
        metadata_path=model_artifact_path.parent.parent.parent
        / "artifacts"
        / "models"
        / "linear_regression_model.metadata.json",
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
    result = ModelTrainingService(training_config).run()
    model_artifact_path.write_bytes(result.model_artifact_path.read_bytes())


def build_inference_dataset() -> str:
    return "\n".join(
        [
            "bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,city,state,zipcode,sale_year,sale_month,sale_day",
            "4,2.5,2500,7000,2,0,0,3,2500,0,1995,0,Renton,WA,98058,2014,5,10",
            "3,1.75,1600,6200,1,0,0,4,1100,500,1955,0,Kent,WA,98031,2014,5,11",
            "",
        ]
    )


def build_training_dataset() -> str:
    return "\n".join(
        [
            "bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,city,state,zipcode,sale_year,sale_month,sale_day,price",
            "3,2,1340,1384,3,0,0,3,1340,0,2008,0,Seattle,WA,98103,2014,5,9,376000",
            "4,3.25,3540,159430,2,0,0,3,3540,0,2007,0,Carnation,WA,98014,2014,5,9,800000",
            "5,6.5,7270,130017,2,0,0,3,6420,850,2010,0,Issaquah,WA,98029,2014,5,9,2238888",
            "3,2.5,3370,7911,1,0,0,3,1670,1700,1968,0,Bellevue,WA,98008,2014,5,9,670000",
            "3,2.25,1710,6622,1,0,0,4,1710,0,1976,0,Redmond,WA,98052,2014,5,9,530000",
            "2,1,900,5000,1,0,0,3,900,0,1940,0,Seattle,WA,98115,2014,5,10,250000",
            "",
        ]
    )
