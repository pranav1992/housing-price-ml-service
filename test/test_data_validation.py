from __future__ import annotations

from pathlib import Path

import pytest

from src.configuration import DataValidationConfig, load_data_validation_config
from src.data_validation import DataValidationService
from src.exceptions import DataValidationError


def test_validation_service_passes_valid_dataset() -> None:
    config = build_validation_config(
        Path("data/raw/usa-housing-dataset/USA Housing Dataset.csv").resolve()
    )

    result = DataValidationService(config).run()

    assert result.row_count == 4140
    assert result.column_count == 18
    assert "Column `price` contains 49 non-positive value(s)." in result.warnings


def test_validation_service_fails_when_required_column_is_missing(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.csv"
    dataset_path.write_text("price,bedrooms\n100,2\n", encoding="utf-8")
    config = build_validation_config(dataset_path)

    with pytest.raises(DataValidationError, match="missing required columns"):
        DataValidationService(config).run()


def test_validation_service_fails_when_numeric_value_is_invalid(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.csv"
    dataset_path.write_text(build_csv(price="not-a-number"), encoding="utf-8")
    config = build_validation_config(dataset_path)

    with pytest.raises(DataValidationError, match="non-numeric value"):
        DataValidationService(config).run()


def test_validation_service_fails_when_duplicate_rows_exist(tmp_path: Path) -> None:
    dataset_path = tmp_path / "duplicate.csv"
    row = build_csv()
    dataset_path.write_text(row + row.split("\n", maxsplit=1)[1], encoding="utf-8")
    config = build_validation_config(dataset_path)

    with pytest.raises(DataValidationError, match="Duplicate row detected"):
        DataValidationService(config).run()


def test_load_data_validation_config_resolves_lists_and_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    config_dir.mkdir()
    config_file = config_dir / "model_config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_validation:",
                "  data_file_path: data/raw/usa-housing-dataset/USA Housing Dataset.csv",
                "  required_columns: date, price, country",
                "  non_empty_columns: date, price, country",
                "  numeric_columns: price",
                "  datetime_columns: date",
                "  strict_positive_columns: sqft_living",
                "  warning_non_positive_columns: price",
                "  expected_constant_values: country=USA",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_data_validation_config(config_file, project_root=project_root)

    assert loaded.data_file_path == project_root / "data" / "raw" / "usa-housing-dataset" / "USA Housing Dataset.csv"
    assert loaded.required_columns == ("date", "price", "country")
    assert loaded.expected_constant_values == (("country", "USA"),)


def build_validation_config(data_file_path: Path) -> DataValidationConfig:
    return DataValidationConfig(
        data_file_path=data_file_path,
        required_columns=(
            "date",
            "price",
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
            "street",
            "city",
            "statezip",
            "country",
        ),
        non_empty_columns=(
            "date",
            "price",
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
            "street",
            "city",
            "statezip",
            "country",
        ),
        numeric_columns=(
            "price",
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
        datetime_columns=("date",),
        strict_positive_columns=("sqft_living", "sqft_lot", "sqft_above", "yr_built"),
        warning_non_positive_columns=("price",),
        expected_constant_values=(("country", "USA"),),
    )


def build_csv(*, price: str = "376000.0") -> str:
    header = (
        "date,price,bedrooms,bathrooms,sqft_living,sqft_lot,floors,waterfront,view,"
        "condition,sqft_above,sqft_basement,yr_built,yr_renovated,street,city,statezip,country\n"
    )
    row = (
        f"2014-05-09 00:00:00,{price},3.0,2.0,1340,1384,3.0,0,0,3,1340,0,2008,0,"
        "9245-9249 Fremont Ave N,Seattle,WA 98103,USA\n"
    )
    return header + row
