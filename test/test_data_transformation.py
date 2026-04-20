from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.configuration import DataTransformationConfig, load_data_transformation_config
from src.data_transformation import DataTransformationService
from src.exceptions import DataTransformationError


def test_transformation_service_writes_cleaned_dataset_and_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    input_path.write_text(build_dataset(), encoding="utf-8")
    config = build_config(tmp_path, input_path)

    result = DataTransformationService(config).run()

    with result.output_path.open("r", encoding="utf-8", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.input_row_count == 2
    assert result.output_row_count == 1
    assert result.dropped_row_count == 1
    assert "street" not in rows[0]
    assert "country" not in rows[0]
    assert rows[0]["state"] == "WA"
    assert rows[0]["zipcode"] == "98103"
    assert rows[0]["sale_year"] == "2014"
    assert rows[0]["sale_month"] == "5"
    assert rows[0]["sale_day"] == "9"
    assert rows[0]["price"] == "376000"
    assert metadata["output_row_count"] == 1
    assert metadata["target_column"] == "price"


def test_transformation_service_fails_on_invalid_date(tmp_path: Path) -> None:
    input_path = tmp_path / "raw.csv"
    input_path.write_text(build_dataset(date="bad-date"), encoding="utf-8")
    config = build_config(tmp_path, input_path)

    with pytest.raises(DataTransformationError, match="invalid value"):
        DataTransformationService(config).run()


def test_load_data_transformation_config_resolves_paths_and_fields(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    config_dir.mkdir()
    config_file = config_dir / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_transformation:",
                "  input_data_path: data/raw/usa-housing-dataset/USA Housing Dataset.csv",
                "  transformed_data_path: artifacts/processed/usa_housing_transformed.csv",
                "  metadata_path: artifacts/processed/usa_housing_transformed.metadata.json",
                "  target_column: price",
                "  date_column: date",
                "  numeric_columns: bedrooms, bathrooms",
                "  categorical_columns: city",
                "  drop_columns: street, country",
                "  statezip_column: statezip",
                "  state_column_name: state",
                "  zipcode_column_name: zipcode",
                "  drop_non_positive_target: true",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_data_transformation_config(config_file, project_root=project_root)

    assert loaded.input_data_path == project_root / "data" / "raw" / "usa-housing-dataset" / "USA Housing Dataset.csv"
    assert loaded.transformed_data_path == project_root / "artifacts" / "processed" / "usa_housing_transformed.csv"
    assert loaded.drop_non_positive_target is True
    assert loaded.drop_columns == ("street", "country")


def build_config(base_dir: Path, input_path: Path) -> DataTransformationConfig:
    return DataTransformationConfig(
        input_data_path=input_path,
        transformed_data_path=base_dir / "artifacts" / "processed" / "usa_housing_transformed.csv",
        metadata_path=base_dir / "artifacts" / "processed" / "usa_housing_transformed.metadata.json",
        target_column="price",
        date_column="date",
        numeric_columns=(
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
        ),
        categorical_columns=("city",),
        drop_columns=("street", "country", "statezip", "date"),
        statezip_column="statezip",
        state_column_name="state",
        zipcode_column_name="zipcode",
        drop_non_positive_target=True,
    )


def build_dataset(*, date: str = "2014-05-09 00:00:00") -> str:
    return "\n".join(
        [
            "date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,city,statezip,country",
            f"{date},376000.0,3.0,2.0,1340,1384,3.0,0,0,3,1340,0,2008,0,9245-9249 Fremont Ave N,Seattle,WA 98103,USA",
            f"{date},0.0,4.0,3.25,3540,159430,2.0,0,0,3,3540,0,2007,0,33001 NE 24th St,Carnation,WA 98014,USA",
            "",
        ]
    )
