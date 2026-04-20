from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from src.configuration import DataTrainingConfig, load_data_training_config
from src.exceptions import ModelTrainingError
from src.model_training import ModelTrainingService


def test_training_service_writes_model_artifact_and_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text(build_training_dataset(), encoding="utf-8")
    config = build_config(tmp_path, input_path)

    result = ModelTrainingService(config).run()

    assert result.model_name == "linear_regression"
    assert result.input_row_count == 6
    assert result.train_row_count == 6
    assert result.test_row_count == 2
    assert result.feature_count > 0
    assert result.model_artifact_path.exists()
    assert result.metadata_path.exists()

    with result.model_artifact_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert payload["model_name"] == "linear_regression"
    assert metadata["train_row_count"] == 6
    assert metadata["test_row_count"] == 2
    assert metadata["target_column"] == "price"
    assert "city=Seattle" in metadata["encoded_feature_names"]


def test_training_service_supports_random_forest_model(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text(build_training_dataset(), encoding="utf-8")
    config = build_config(tmp_path, input_path, model_name="random_forest")

    result = ModelTrainingService(config).run()

    with result.model_artifact_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)

    assert result.model_name == "random_forest"
    assert payload["model_name"] == "random_forest"
    assert result.model_artifact_path.name == "random_forest_model.pkl"


def test_training_service_supports_hist_gradient_boosting_model(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text(build_training_dataset(), encoding="utf-8")
    config = build_config(tmp_path, input_path, model_name="hist_gradient_boosting")

    result = ModelTrainingService(config).run()

    with result.model_artifact_path.open("rb") as file_obj:
        payload = pickle.load(file_obj)

    assert result.model_name == "hist_gradient_boosting"
    assert payload["model_name"] == "hist_gradient_boosting"
    assert result.model_artifact_path.name == "hist_gradient_boosting_model.pkl"


def test_training_service_fails_when_required_column_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text("bedrooms,price\n3,100000\n4,120000\n", encoding="utf-8")
    config = build_config(tmp_path, input_path)

    with pytest.raises(ModelTrainingError, match="missing required columns"):
        ModelTrainingService(config).run()


def test_load_data_training_config_resolves_paths_and_fields(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    config_dir.mkdir()
    config_file = config_dir / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_training:",
                "  input_data_path: artifacts/splits/usa_housing_train.csv",
                "  split_metadata_path: artifacts/splits/usa_housing_split.metadata.json",
                "  model_artifact_path: artifacts/models/linear_regression_model.pkl",
                "  metadata_path: artifacts/models/linear_regression_model.metadata.json",
                "  model_name: linear_regression",
                "  target_column: price",
                "  numeric_feature_columns: bedrooms, bathrooms, sqft_living",
                "  categorical_feature_columns: city, state",
                "  test_size: 0.2",
                "  random_state: 42",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_data_training_config(config_file, project_root=project_root)

    assert loaded.input_data_path == project_root / "artifacts" / "splits" / "usa_housing_train.csv"
    assert loaded.split_metadata_path == project_root / "artifacts" / "splits" / "usa_housing_split.metadata.json"
    assert loaded.model_artifact_path == project_root / "artifacts" / "models" / "linear_regression_model.pkl"
    assert loaded.numeric_feature_columns == ("bedrooms", "bathrooms", "sqft_living")
    assert loaded.categorical_feature_columns == ("city", "state")
    assert loaded.test_size == 0.2
    assert loaded.random_state == 42


def build_config(
    base_dir: Path,
    input_path: Path,
    *,
    model_name: str = "linear_regression",
) -> DataTrainingConfig:
    split_metadata_path = base_dir / "artifacts" / "splits" / "usa_housing_split.metadata.json"
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
    return DataTrainingConfig(
        input_data_path=input_path,
        split_metadata_path=split_metadata_path,
        model_artifact_path=base_dir / "artifacts" / "models" / f"{model_name}_model.pkl",
        metadata_path=base_dir / "artifacts" / "models" / f"{model_name}_model.metadata.json",
        model_name=model_name,
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
