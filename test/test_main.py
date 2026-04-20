from __future__ import annotations

from dataclasses import dataclass

import main as main_module


@dataclass
class FakeResult:
    status: str
    sha256: str


class FakeService:
    def __init__(self, config) -> None:
        self.config = config

    def run(self) -> FakeResult:
        return FakeResult(status="already_available", sha256="abc123")


class FakeDownloadedService(FakeService):
    def run(self) -> FakeResult:
        return FakeResult(status="downloaded", sha256="def456")


def test_main_prints_already_available_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeService)

    main_module.main()

    assert capsys.readouterr().out.strip() == "Data is already available with hash: abc123"


def test_main_prints_downloaded_message(monkeypatch, capsys) -> None:
    monkeypatch.setattr(main_module, "load_data_ingestion_config", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "DataIngestionService", FakeDownloadedService)

    main_module.main()

    assert capsys.readouterr().out.strip() == "Data downloaded successfully with hash: def456"
