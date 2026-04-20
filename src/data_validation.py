"""Dataset validation services for the housing pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.configuration import DataValidationConfig
from src.exceptions import DataValidationError


@dataclass(frozen=True, slots=True)
class DataValidationResult:
    """Outcome of a validation run."""

    row_count: int
    column_count: int
    warnings: tuple[str, ...]


class DataValidationService:
    """Validate the ingested CSV before transformation or training."""

    def __init__(self, config: DataValidationConfig) -> None:
        self._config = config

    def run(self) -> DataValidationResult:
        file_path = self._config.data_file_path
        if not file_path.exists():
            raise DataValidationError(f"Validation data file not found: {file_path}")

        with file_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            fieldnames = tuple(reader.fieldnames or ())
            self._validate_columns(fieldnames)

            rows = list(reader)

        if not rows:
            raise DataValidationError(f"Validation data file is empty: {file_path}")

        self._validate_duplicates(rows, fieldnames)
        self._validate_non_empty_columns(rows)
        self._validate_numeric_columns(rows)
        self._validate_datetime_columns(rows)
        self._validate_strict_positive_columns(rows)
        warnings = self._collect_warnings(rows)

        return DataValidationResult(
            row_count=len(rows),
            column_count=len(fieldnames),
            warnings=tuple(warnings),
        )

    def _validate_columns(self, fieldnames: tuple[str, ...]) -> None:
        missing_columns = [column for column in self._config.required_columns if column not in fieldnames]
        if missing_columns:
            raise DataValidationError(
                "Dataset is missing required columns: " + ", ".join(sorted(missing_columns))
            )

    def _validate_duplicates(self, rows: list[dict[str, str]], fieldnames: tuple[str, ...]) -> None:
        seen: set[tuple[str, ...]] = set()
        for index, row in enumerate(rows, start=2):
            marker = tuple(row.get(column, "") for column in fieldnames)
            if marker in seen:
                raise DataValidationError(f"Duplicate row detected at line {index}.")
            seen.add(marker)

    def _validate_non_empty_columns(self, rows: list[dict[str, str]]) -> None:
        for column in self._config.non_empty_columns:
            empty_count = sum(1 for row in rows if not row.get(column, "").strip())
            if empty_count:
                raise DataValidationError(
                    f"Column `{column}` contains {empty_count} empty value(s)."
                )

    def _validate_numeric_columns(self, rows: list[dict[str, str]]) -> None:
        for column in self._config.numeric_columns:
            for index, row in enumerate(rows, start=2):
                value = row.get(column, "").strip()
                try:
                    float(value)
                except ValueError as exc:
                    raise DataValidationError(
                        f"Column `{column}` contains a non-numeric value at line {index}: `{value}`."
                    ) from exc

    def _validate_datetime_columns(self, rows: list[dict[str, str]]) -> None:
        for column in self._config.datetime_columns:
            for index, row in enumerate(rows, start=2):
                value = row.get(column, "").strip()
                try:
                    datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                except ValueError as exc:
                    raise DataValidationError(
                        f"Column `{column}` contains an invalid datetime at line {index}: `{value}`."
                    ) from exc

    def _validate_strict_positive_columns(self, rows: list[dict[str, str]]) -> None:
        for column in self._config.strict_positive_columns:
            invalid_count = sum(1 for row in rows if float(row[column]) <= 0)
            if invalid_count:
                raise DataValidationError(
                    f"Column `{column}` contains {invalid_count} non-positive value(s)."
                )

    def _collect_warnings(self, rows: list[dict[str, str]]) -> list[str]:
        warnings: list[str] = []

        for column in self._config.warning_non_positive_columns:
            invalid_count = sum(1 for row in rows if float(row[column]) <= 0)
            if invalid_count:
                warnings.append(
                    f"Column `{column}` contains {invalid_count} non-positive value(s)."
                )

        for column, expected_value in self._config.expected_constant_values:
            mismatch_count = sum(1 for row in rows if row.get(column, "").strip() != expected_value)
            if mismatch_count:
                warnings.append(
                    f"Column `{column}` contains {mismatch_count} value(s) different from `{expected_value}`."
                )

        return warnings
