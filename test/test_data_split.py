from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.configuration import DataSplitConfig, load_data_split_config
from src.data_split import DataSplitService
from src.exceptions import DataSplitError


def test_split_service_writes_train_test_artifacts_and_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text(build_dataset(), encoding="utf-8")
    config = build_config(tmp_path, input_path)

    result = DataSplitService(config).run()

    with result.train_data_path.open("r", encoding="utf-8", newline="") as file_obj:
        train_rows = list(csv.DictReader(file_obj))
    with result.test_data_path.open("r", encoding="utf-8", newline="") as file_obj:
        test_rows = list(csv.DictReader(file_obj))
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert result.input_row_count == 8
    assert result.train_row_count == 6
    assert result.test_row_count == 2
    assert len(train_rows) == 6
    assert len(test_rows) == 2
    assert metadata["target_column"] == "price"
    assert metadata["train_row_count"] == 6
    assert metadata["test_row_count"] == 2


def test_split_service_fails_when_target_column_is_missing(tmp_path: Path) -> None:
    input_path = tmp_path / "processed.csv"
    input_path.write_text("bedrooms,city\n3,Seattle\n4,Redmond\n", encoding="utf-8")
    config = build_config(tmp_path, input_path)

    with pytest.raises(DataSplitError, match="missing target column"):
        DataSplitService(config).run()


def test_load_data_split_config_resolves_paths_and_fields(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    config_dir.mkdir()
    config_file = config_dir / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_split:",
                "  input_data_path: artifacts/processed/usa_housing_transformed.csv",
                "  train_data_path: artifacts/splits/usa_housing_train.csv",
                "  test_data_path: artifacts/splits/usa_housing_test.csv",
                "  metadata_path: artifacts/splits/usa_housing_split.metadata.json",
                "  target_column: price",
                "  test_size: 0.2",
                "  random_state: 42",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_data_split_config(config_file, project_root=project_root)

    assert loaded.input_data_path == project_root / "artifacts" / "processed" / "usa_housing_transformed.csv"
    assert loaded.train_data_path == project_root / "artifacts" / "splits" / "usa_housing_train.csv"
    assert loaded.test_data_path == project_root / "artifacts" / "splits" / "usa_housing_test.csv"
    assert loaded.metadata_path == project_root / "artifacts" / "splits" / "usa_housing_split.metadata.json"
    assert loaded.target_column == "price"
    assert loaded.test_size == 0.2
    assert loaded.random_state == 42


def build_config(base_dir: Path, input_path: Path) -> DataSplitConfig:
    return DataSplitConfig(
        input_data_path=input_path,
        train_data_path=base_dir / "artifacts" / "splits" / "usa_housing_train.csv",
        test_data_path=base_dir / "artifacts" / "splits" / "usa_housing_test.csv",
        metadata_path=base_dir / "artifacts" / "splits" / "usa_housing_split.metadata.json",
        target_column="price",
        test_size=0.25,
        random_state=7,
    )


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
