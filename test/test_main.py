from __future__ import annotations

from dataclasses import dataclass

import main as main_module


@dataclass
class FakeResult:
    status: str
    sha256: str


@dataclass
class FakeValidationResult:
    row_count: int
    column_count: int
    warnings: tuple[str, ...]


@dataclass
class FakeTransformationResult:
    output_path: str
    output_row_count: int


@dataclass
class FakeSplitResult:
    metadata_path: str
    train_row_count: int
    test_row_count: int


@dataclass
class FakeTrainingResult:
    model_name: str
    model_artifact_path: str
    train_row_count: int


@dataclass
class FakeEvaluationResult:
    mae: float
    rmse: float
    r2: float


@dataclass
class FakeInferenceResult:
    predictions_path: str
    predicted_row_count: int


class FakeService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeResult:
        return FakeResult(status="already_available", sha256="abc123")


class FakeDownloadedService(FakeService):
    def run(self) -> FakeResult:
        return FakeResult(status="downloaded", sha256="def456")


class FakeValidationService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeValidationResult:
        return FakeValidationResult(row_count=4140, column_count=18, warnings=())


class FakeWarningValidationService(FakeValidationService):
    def run(self) -> FakeValidationResult:
        return FakeValidationResult(
            row_count=4140,
            column_count=18,
            warnings=("Column `price` contains 49 non-positive value(s).",),
        )


class FakeTransformationService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeTransformationResult:
        return FakeTransformationResult(
            output_path="artifacts/processed/usa_housing_transformed.csv",
            output_row_count=4091,
        )


class FakeSplitService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeSplitResult:
        return FakeSplitResult(
            metadata_path="artifacts/splits/usa_housing_split.metadata.json",
            train_row_count=3272,
            test_row_count=819,
        )


class FakeTrainingService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeTrainingResult:
        return FakeTrainingResult(
            model_name="linear_regression",
            model_artifact_path="artifacts/models/linear_regression_model.pkl",
            train_row_count=3272,
        )


class FakeEvaluationService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeEvaluationResult:
        return FakeEvaluationResult(mae=12345.67, rmse=45678.9, r2=0.8123)


class FakeInferenceService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeInferenceResult:
        return FakeInferenceResult(
            predictions_path="artifacts/predictions/linear_regression_predictions.csv",
            predicted_row_count=819,
        )


def test_main_prints_already_available_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_evaluation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_inference_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_split_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_training_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_validation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_transformation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeService)
    monkeypatch.setattr(main_module, "DataValidationService", FakeValidationService)
    monkeypatch.setattr(main_module, "DataTransformationService", FakeTransformationService)
    monkeypatch.setattr(main_module, "DataSplitService", FakeSplitService)
    monkeypatch.setattr(main_module, "ModelTrainingService", FakeTrainingService)
    monkeypatch.setattr(main_module, "ModelEvaluationService", FakeEvaluationService)
    monkeypatch.setattr(main_module, "ModelInferenceService", FakeInferenceService)

    main_module.main()

    assert capsys.readouterr().out.strip() == (
        "Data is already available with hash: abc123\n"
        "Data validation passed for 4140 rows and 18 columns.\n"
        "Data transformation completed: 4091 rows written to "
        "artifacts/processed/usa_housing_transformed.csv\n"
        "Data split completed: 3272 train rows and 819 test rows written to "
        "artifacts/splits/usa_housing_split.metadata.json\n"
        "Model training completed: linear_regression fitted on 3272 rows and saved to "
        "artifacts/models/linear_regression_model.pkl\n"
        "Model evaluation completed: MAE=12345.67, RMSE=45678.90, R2=0.8123\n"
        "Model inference completed: 819 rows written to "
        "artifacts/predictions/linear_regression_predictions.csv"
    )


def test_main_prints_downloaded_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_evaluation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_inference_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_split_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_training_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_validation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_transformation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeDownloadedService)
    monkeypatch.setattr(main_module, "DataValidationService", FakeValidationService)
    monkeypatch.setattr(main_module, "DataTransformationService", FakeTransformationService)
    monkeypatch.setattr(main_module, "DataSplitService", FakeSplitService)
    monkeypatch.setattr(main_module, "ModelTrainingService", FakeTrainingService)
    monkeypatch.setattr(main_module, "ModelEvaluationService", FakeEvaluationService)
    monkeypatch.setattr(main_module, "ModelInferenceService", FakeInferenceService)

    main_module.main()

    assert capsys.readouterr().out.strip() == (
        "Data downloaded successfully with hash: def456\n"
        "Data validation passed for 4140 rows and 18 columns.\n"
        "Data transformation completed: 4091 rows written to "
        "artifacts/processed/usa_housing_transformed.csv\n"
        "Data split completed: 3272 train rows and 819 test rows written to "
        "artifacts/splits/usa_housing_split.metadata.json\n"
        "Model training completed: linear_regression fitted on 3272 rows and saved to "
        "artifacts/models/linear_regression_model.pkl\n"
        "Model evaluation completed: MAE=12345.67, RMSE=45678.90, R2=0.8123\n"
        "Model inference completed: 819 rows written to "
        "artifacts/predictions/linear_regression_predictions.csv"
    )


def test_main_prints_validation_warnings(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_evaluation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_inference_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_split_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_training_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_validation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_transformation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeService)
    monkeypatch.setattr(main_module, "DataValidationService", FakeWarningValidationService)
    monkeypatch.setattr(main_module, "DataTransformationService", FakeTransformationService)
    monkeypatch.setattr(main_module, "DataSplitService", FakeSplitService)
    monkeypatch.setattr(main_module, "ModelTrainingService", FakeTrainingService)
    monkeypatch.setattr(main_module, "ModelEvaluationService", FakeEvaluationService)
    monkeypatch.setattr(main_module, "ModelInferenceService", FakeInferenceService)

    main_module.main()

    assert capsys.readouterr().out.strip() == (
        "Data is already available with hash: abc123\n"
        "Data validation passed with warnings: "
        "Column `price` contains 49 non-positive value(s).\n"
        "Data transformation completed: 4091 rows written to "
        "artifacts/processed/usa_housing_transformed.csv\n"
        "Data split completed: 3272 train rows and 819 test rows written to "
        "artifacts/splits/usa_housing_split.metadata.json\n"
        "Model training completed: linear_regression fitted on 3272 rows and saved to "
        "artifacts/models/linear_regression_model.pkl\n"
        "Model evaluation completed: MAE=12345.67, RMSE=45678.90, R2=0.8123\n"
        "Model inference completed: 819 rows written to "
        "artifacts/predictions/linear_regression_predictions.csv"
    )
