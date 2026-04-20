from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.configuration import DataIngestionConfig, load_data_ingestion_config
from src.data_ingestion import DataIngestionService
from src.exceptions import DataIngestionError


class FakeDownloader:
    def __init__(self, source_file: Path | None = None, *, error: Exception | None = None) -> None:
        self.source_file = source_file
        self.error = error
        self.calls = 0

    def download(self, source_url: str, destination_path: Path) -> Path:
        self.calls += 1
        if self.error:
            raise self.error

        assert self.source_file is not None
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        destination_path.write_bytes(self.source_file.read_bytes())
        return destination_path


def test_existing_file_skips_download_and_returns_hash(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    config.downloaded_file_path.parent.mkdir(parents=True, exist_ok=True)
    config.downloaded_file_path.write_text("price,rooms\n100,2\n", encoding="utf-8")
    downloader = FakeDownloader()

    result = DataIngestionService(config, downloader=downloader).run()

    assert result.status == "already_available"
    assert downloader.calls == 0
    assert result.file_path == config.downloaded_file_path
    assert len(result.sha256) == 64
    assert config.manifest_path.exists()


def test_missing_file_downloads_and_writes_manifest(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    source_file = tmp_path / "source.csv"
    source_file.write_text("price,rooms\n100,2\n", encoding="utf-8")
    downloader = FakeDownloader(source_file)

    result = DataIngestionService(config, downloader=downloader).run()
    manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))

    assert result.status == "downloaded"
    assert downloader.calls == 1
    assert config.cache_file_path.exists()
    assert config.downloaded_file_path.exists()
    assert manifest["source_url"] == config.source_url
    assert manifest["sha256"] == result.sha256


def test_existing_cache_materializes_output_without_redownload(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    config.cache_file_path.parent.mkdir(parents=True, exist_ok=True)
    config.cache_file_path.write_text("price,rooms\n100,2\n", encoding="utf-8")
    downloader = FakeDownloader()

    result = DataIngestionService(config, downloader=downloader).run()

    assert result.status == "already_available"
    assert downloader.calls == 0
    assert config.downloaded_file_path.exists()
    assert result.file_path == config.downloaded_file_path


def test_download_failure_is_clear(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    downloader = FakeDownloader(error=DataIngestionError("download failed"))

    with pytest.raises(DataIngestionError, match="download failed"):
        DataIngestionService(config, downloader=downloader).run()


def test_zero_byte_file_is_not_marked_available(tmp_path: Path) -> None:
    config = build_config(tmp_path)
    config.downloaded_file_path.parent.mkdir(parents=True, exist_ok=True)
    config.downloaded_file_path.write_bytes(b"")
    downloader = FakeDownloader(error=DataIngestionError("download failed"))

    with pytest.raises(DataIngestionError, match="download failed"):
        DataIngestionService(config, downloader=downloader).run()


def test_load_data_ingestion_config_resolves_project_relative_paths(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config_dir = project_root / "config"
    config_dir.mkdir()
    config_file = config_dir / "config.yaml"
    config_file.write_text(
        "\n".join(
            [
                "data_ingestion:",
                "  source_name: huggingface",
                "  source_url: https://huggingface.co/datasets/Nhule0502/USA_house_price/resolve/main/USA%20Housing%20Dataset.csv",
                "  file_name: USA Housing Dataset.csv",
                "  raw_data_dir: data/raw",
                "  extracted_data_dir: data/raw/usa-housing-dataset",
                "  manifest_path: data/raw/usa-housing-dataset.manifest.json",
            ]
        ),
        encoding="utf-8",
    )

    loaded = load_data_ingestion_config(config_file, project_root=project_root)

    assert loaded.raw_data_dir == project_root / "data" / "raw"
    assert loaded.extracted_data_dir == project_root / "data" / "raw" / "usa-housing-dataset"
    assert loaded.file_name == "USA Housing Dataset.csv"


def build_config(base_dir: Path) -> DataIngestionConfig:
    return DataIngestionConfig(
        source_name="huggingface",
        source_url="https://huggingface.co/datasets/Nhule0502/USA_house_price/resolve/main/USA%20Housing%20Dataset.csv",
        raw_data_dir=base_dir / "data" / "raw",
        extracted_data_dir=base_dir / "data" / "raw" / "usa-housing-dataset",
        manifest_path=base_dir / "data" / "raw" / "usa-housing-dataset.manifest.json",
        file_name="USA Housing Dataset.csv",
    )
