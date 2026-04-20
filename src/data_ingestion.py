"""Production-oriented dataset ingestion pipeline."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from src.configuration import DataIngestionConfig
from src.exceptions import DataIngestionError

logger = logging.getLogger(__name__)


class DatasetDownloader(Protocol):
    """Downloader contract used by the ingestion service."""

    def download(self, source_url: str, destination_path: Path) -> Path:
        """Download a dataset file and return its final path."""


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Outcome of a data ingestion run."""

    status: str
    file_path: Path
    sha256: str
    manifest_path: Path


class HttpFileDownloader:
    """Downloader backed by HTTP for a single dataset file."""

    def download(self, source_url: str, destination_path: Path) -> Path:
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = destination_path.with_suffix(destination_path.suffix + ".part")

        try:
            with urlopen(source_url, timeout=120) as response, temp_path.open("wb") as file_obj:
                while chunk := response.read(1024 * 1024):
                    file_obj.write(chunk)
        except (HTTPError, URLError, TimeoutError) as exc:
            temp_path.unlink(missing_ok=True)
            raise DataIngestionError(f"Failed to download dataset file from `{source_url}`.") from exc

        if temp_path.stat().st_size <= 0:
            temp_path.unlink(missing_ok=True)
            raise DataIngestionError(f"Downloaded file from `{source_url}` is empty.")

        temp_path.replace(destination_path)
        return destination_path


class DataIngestionService:
    """Coordinates presence checks, download, file materialization, and manifest updates."""

    def __init__(
        self,
        config: DataIngestionConfig,
        *,
        downloader: DatasetDownloader | None = None,
    ) -> None:
        self._config = config
        self._downloader = downloader or HttpFileDownloader()

    def run(self) -> IngestionResult:
        self._ensure_directories()

        file_path = self._resolve_existing_file()
        status = "already_available"

        if file_path is not None:
            logger.info("Found dataset file at %s", file_path)
            if not self._is_valid_file(file_path):
                logger.warning("Dataset file at %s is invalid. Triggering a fresh download.", file_path)
                file_path.unlink(missing_ok=True)
                file_path = None

        if file_path is None:
            status = "downloaded"
            file_path = self._download_and_materialize()

        if not self._is_valid_file(file_path):
            raise DataIngestionError(
                f"Dataset file `{file_path}` is corrupt or empty. Remove it and rerun ingestion."
            )

        sha256 = self._compute_sha256(file_path)
        self._write_manifest(file_path=file_path, sha256=sha256)

        return IngestionResult(
            status=status,
            file_path=file_path,
            sha256=sha256,
            manifest_path=self._config.manifest_path,
        )

    def _ensure_directories(self) -> None:
        self._config.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self._config.extracted_data_dir.mkdir(parents=True, exist_ok=True)
        self._config.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    def _resolve_existing_file(self) -> Path | None:
        if self._config.downloaded_file_path.exists():
            return self._config.downloaded_file_path

        if self._config.cache_file_path.exists():
            self._materialize_file(self._config.cache_file_path)
            return self._config.downloaded_file_path

        manifest_file = self._file_from_manifest()
        if manifest_file and manifest_file.exists():
            return manifest_file

        return None

    def _file_from_manifest(self) -> Path | None:
        manifest_path = self._config.manifest_path
        if not manifest_path.exists():
            return None

        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Manifest `%s` is invalid JSON. Ignoring it.", manifest_path)
            return None

        manifest_source_url = manifest_data.get("source_url")
        file_path = manifest_data.get("file_path")
        if manifest_source_url != self._config.source_url or not isinstance(file_path, str):
            return None

        return Path(file_path)

    def _download_and_materialize(self) -> Path:
        cached_file_path = self._downloader.download(self._config.source_url, self._config.cache_file_path)
        logger.info("Downloaded dataset file to %s", cached_file_path)
        self._materialize_file(cached_file_path)
        return self._config.downloaded_file_path

    def _materialize_file(self, source_path: Path) -> None:
        if not source_path.exists():
            raise DataIngestionError(f"Expected source file `{source_path}` to exist before materialization.")
        self._config.downloaded_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, self._config.downloaded_file_path)

    @staticmethod
    def _is_valid_file(file_path: Path) -> bool:
        return file_path.exists() and file_path.stat().st_size > 0

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        digest = hashlib.sha256()
        with file_path.open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _write_manifest(self, *, file_path: Path, sha256: str) -> None:
        manifest_payload = {
            "source_name": self._config.source_name,
            "source_url": self._config.source_url,
            "file_name": self._config.file_name,
            "file_path": str(file_path),
            "cached_file_path": str(self._config.cache_file_path),
            "sha256": sha256,
            "downloaded_at": datetime.now(UTC).isoformat(),
        }

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._config.manifest_path.parent,
            delete=False,
        ) as file_obj:
            json.dump(manifest_payload, file_obj, indent=2)
            file_obj.write("\n")
            temp_name = file_obj.name

        Path(temp_name).replace(self._config.manifest_path)
