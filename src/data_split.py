"""Deterministic train/test split services for the housing pipeline."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.configuration import DataSplitConfig
from src.exceptions import DataSplitError


@dataclass(frozen=True, slots=True)
class DataSplitResult:
    """Outcome of a deterministic split run."""

    train_data_path: Path
    test_data_path: Path
    metadata_path: Path
    input_row_count: int
    train_row_count: int
    test_row_count: int
    target_column: str


class DataSplitService:
    """Persist a deterministic train/test split from transformed tabular data."""

    def __init__(self, config: DataSplitConfig) -> None:
        self._config = config

    def run(self) -> DataSplitResult:
        input_path = self._config.input_data_path
        if not input_path.exists():
            raise DataSplitError(f"Split input file not found: {input_path}")

        rows, fieldnames = self._read_rows(input_path)
        if self._config.target_column not in fieldnames:
            raise DataSplitError(
                f"Split input is missing target column `{self._config.target_column}`."
            )

        train_test_split = self._load_train_test_split()
        train_rows, test_rows = train_test_split(
            rows,
            test_size=self._config.test_size,
            random_state=self._config.random_state,
        )

        if not train_rows or not test_rows:
            raise DataSplitError("Split configuration produced an empty train or test set.")

        self._config.train_data_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.test_data_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_csv(self._config.train_data_path, fieldnames, train_rows)
        self._write_csv(self._config.test_data_path, fieldnames, test_rows)
        self._write_metadata(
            input_row_count=len(rows),
            train_row_count=len(train_rows),
            test_row_count=len(test_rows),
            fieldnames=fieldnames,
        )

        return DataSplitResult(
            train_data_path=self._config.train_data_path,
            test_data_path=self._config.test_data_path,
            metadata_path=self._config.metadata_path,
            input_row_count=len(rows),
            train_row_count=len(train_rows),
            test_row_count=len(test_rows),
            target_column=self._config.target_column,
        )

    @staticmethod
    def _read_rows(input_path: Path) -> tuple[list[dict[str, str]], tuple[str, ...]]:
        with input_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = tuple(reader.fieldnames or ())
        if not rows:
            raise DataSplitError(f"Split input file is empty: {input_path}")
        return rows, fieldnames

    @staticmethod
    def _load_train_test_split():
        try:
            from sklearn.model_selection import train_test_split
        except ImportError as exc:
            raise DataSplitError(
                "scikit-learn is required for data splitting. Install project dependencies before running the split stage."
            ) from exc
        return train_test_split

    @staticmethod
    def _write_csv(
        output_path: Path,
        fieldnames: tuple[str, ...],
        rows: list[dict[str, str]],
    ) -> None:
        with output_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(fieldnames))
            writer.writeheader()
            writer.writerows(rows)

    def _write_metadata(
        self,
        *,
        input_row_count: int,
        train_row_count: int,
        test_row_count: int,
        fieldnames: tuple[str, ...],
    ) -> None:
        metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_data_path": str(self._config.input_data_path),
            "train_data_path": str(self._config.train_data_path),
            "test_data_path": str(self._config.test_data_path),
            "input_row_count": input_row_count,
            "train_row_count": train_row_count,
            "test_row_count": test_row_count,
            "target_column": self._config.target_column,
            "columns": list(fieldnames),
            "test_size": self._config.test_size,
            "random_state": self._config.random_state,
        }

        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=self._config.metadata_path.parent,
            delete=False,
        ) as file_obj:
            json.dump(metadata, file_obj, indent=2)
            file_obj.write("\n")
            temp_name = file_obj.name

        Path(temp_name).replace(self._config.metadata_path)
