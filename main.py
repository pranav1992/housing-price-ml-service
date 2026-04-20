from pathlib import Path

from src.configuration import load_data_ingestion_config, load_data_validation_config
from src.data_ingestion import DataIngestionService
from src.data_validation import DataValidationService


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_data_ingestion_config(project_root / "config" / "config.yaml", project_root=project_root)
    result = DataIngestionService(config).run()
    validation_config = load_data_validation_config(
        project_root / "config" / "model_config.yaml",
        project_root=project_root,
    )
    validation_result = DataValidationService(validation_config).run()

    if result.status == "already_available":
        print(f"Data is already available with hash: {result.sha256}")
    else:
        print(f"Data downloaded successfully with hash: {result.sha256}")

    if validation_result.warnings:
        print(
            "Data validation passed with warnings: "
            + "; ".join(validation_result.warnings)
        )
        return

    print(
        f"Data validation passed for {validation_result.row_count} rows "
        f"and {validation_result.column_count} columns."
    )


if __name__ == "__main__":
    main()
