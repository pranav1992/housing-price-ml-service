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


def test_main_prints_already_available_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_validation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeService)
    monkeypatch.setattr(main_module, "DataValidationService", FakeValidationService)

    main_module.main()

    assert capsys.readouterr().out.strip() == (
        "Data is already available with hash: abc123\n"
        "Data validation passed for 4140 rows and 18 columns."
    )


def test_main_prints_downloaded_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_validation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeDownloadedService)
    monkeypatch.setattr(main_module, "DataValidationService", FakeValidationService)

    main_module.main()

    assert capsys.readouterr().out.strip() == (
        "Data downloaded successfully with hash: def456\n"
        "Data validation passed for 4140 rows and 18 columns."
    )


def test_main_prints_validation_warnings(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "load_data_validation_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeService)
    monkeypatch.setattr(main_module, "DataValidationService", FakeWarningValidationService)

    main_module.main()

    assert capsys.readouterr().out.strip() == (
        "Data is already available with hash: abc123\n"
        "Data validation passed with warnings: "
        "Column `price` contains 49 non-positive value(s)."
    )
