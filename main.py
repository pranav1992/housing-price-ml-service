from pathlib import Path

from src.configuration import (
    load_data_evaluation_config,
    load_data_ingestion_config,
    load_data_training_config,
    load_data_transformation_config,
    load_data_validation_config,
)
from src.data_ingestion import DataIngestionService
from src.data_transformation import DataTransformationService
from src.data_validation import DataValidationService
from src.model_evaluation import ModelEvaluationService
from src.model_training import ModelTrainingService


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_data_ingestion_config(project_root / "config" / "config.yaml", project_root=project_root)
    result = DataIngestionService(config).run()
    validation_config = load_data_validation_config(
        project_root / "config" / "model_config.yaml",
        project_root=project_root,
    )
    validation_result = DataValidationService(validation_config).run()
    transformation_config = load_data_transformation_config(
        project_root / "config" / "model_config.yaml",
        project_root=project_root,
    )
    transformation_result = DataTransformationService(transformation_config).run()
    training_config = load_data_training_config(
        project_root / "config" / "model_config.yaml",
        project_root=project_root,
    )
    training_result = ModelTrainingService(training_config).run()
    evaluation_config = load_data_evaluation_config(
        project_root / "config" / "model_config.yaml",
        project_root=project_root,
    )
    evaluation_result = ModelEvaluationService(evaluation_config).run()

    if result.status == "already_available":
        print(f"Data is already available with hash: {result.sha256}")
    else:
        print(f"Data downloaded successfully with hash: {result.sha256}")

    if validation_result.warnings:
        print(
            "Data validation passed with warnings: "
            + "; ".join(validation_result.warnings)
        )
    else:
        print(
            f"Data validation passed for {validation_result.row_count} rows "
            f"and {validation_result.column_count} columns."
        )

    print(
        f"Data transformation completed: {transformation_result.output_row_count} rows written to "
        f"{transformation_result.output_path}"
    )
    print(
        f"Model training completed: {training_result.model_name} fitted on "
        f"{training_result.train_row_count} rows and saved to {training_result.model_artifact_path}"
    )
    print(
        f"Model evaluation completed: MAE={evaluation_result.mae:.2f}, "
        f"RMSE={evaluation_result.rmse:.2f}, R2={evaluation_result.r2:.4f}"
    )


if __name__ == "__main__":
    main()
