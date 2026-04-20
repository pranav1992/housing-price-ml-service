"""Data transformation services for the housing pipeline."""

from __future__ import annotations

import csv
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from src.configuration import DataTransformationConfig
from src.exceptions import DataTransformationError


@dataclass(frozen=True, slots=True)
class DataTransformationResult:
    """Outcome of a transformation run."""

    output_path: Path
    metadata_path: Path
    input_row_count: int
    output_row_count: int
    dropped_row_count: int
    feature_columns: tuple[str, ...]
    target_column: str


class DataTransformationService:
    """Transform validated raw data into model-ready tabular artifacts."""

    def __init__(self, config: DataTransformationConfig) -> None:
        self._config = config

    def run(self) -> DataTransformationResult:
        input_path = self._config.input_data_path
        if not input_path.exists():
            raise DataTransformationError(f"Transformation input file not found: {input_path}")

        self._config.transformed_data_path.parent.mkdir(parents=True, exist_ok=True)
        self._config.metadata_path.parent.mkdir(parents=True, exist_ok=True)

        with input_path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            rows = list(reader)
            fieldnames = tuple(reader.fieldnames or ())

        if not rows:
            raise DataTransformationError(f"Transformation input file is empty: {input_path}")

        self._validate_required_columns(fieldnames)

        transformed_rows = []
        dropped_row_count = 0
        output_columns = self._build_output_columns()

        for index, row in enumerate(rows, start=2):
            transformed_row = self._transform_row(row, index=index)
            if transformed_row is None:
                dropped_row_count += 1
                continue
            transformed_rows.append(transformed_row)

        if not transformed_rows:
            raise DataTransformationError("All rows were dropped during transformation.")

        self._write_transformed_csv(output_columns, transformed_rows)
        self._write_metadata(
            input_row_count=len(rows),
            output_row_count=len(transformed_rows),
            dropped_row_count=dropped_row_count,
            output_columns=output_columns,
        )

        return DataTransformationResult(
            output_path=self._config.transformed_data_path,
            metadata_path=self._config.metadata_path,
            input_row_count=len(rows),
            output_row_count=len(transformed_rows),
            dropped_row_count=dropped_row_count,
            feature_columns=tuple(
                column for column in output_columns if column != self._config.target_column
            ),
            target_column=self._config.target_column,
        )

    def _validate_required_columns(self, fieldnames: tuple[str, ...]) -> None:
        required = {
            self._config.target_column,
            self._config.date_column,
            self._config.statezip_column,
            *self._config.numeric_columns,
            *self._config.categorical_columns,
        }
        missing = sorted(column for column in required if column not in fieldnames)
        if missing:
            raise DataTransformationError(
                "Transformation input is missing required columns: " + ", ".join(missing)
            )

    def _build_output_columns(self) -> tuple[str, ...]:
        columns = [
            *self._config.numeric_columns,
            *self._config.categorical_columns,
            self._config.state_column_name,
            self._config.zipcode_column_name,
            "sale_year",
            "sale_month",
            "sale_day",
            self._config.target_column,
        ]

        deduped = []
        seen = set()
        for column in columns:
            if column in self._config.drop_columns:
                continue
            if column not in seen:
                deduped.append(column)
                seen.add(column)
        return tuple(deduped)

    def _transform_row(self, row: dict[str, str], *, index: int) -> dict[str, str] | None:
        target_value_raw = row.get(self._config.target_column, "").strip()
        try:
            target_value = float(target_value_raw)
        except ValueError as exc:
            raise DataTransformationError(
                f"Target column `{self._config.target_column}` contains a non-numeric value at line {index}: "
                f"`{target_value_raw}`."
            ) from exc

        if self._config.drop_non_positive_target and target_value <= 0:
            return None

        date_raw = row.get(self._config.date_column, "").strip()
        try:
            parsed_date = datetime.strptime(date_raw, "%Y-%m-%d %H:%M:%S")
        except ValueError as exc:
            raise DataTransformationError(
                f"Date column `{self._config.date_column}` contains an invalid value at line {index}: "
                f"`{date_raw}`."
            ) from exc

        state_raw = row.get(self._config.statezip_column, "").strip()
        state, zipcode = self._split_statezip(state_raw)

        transformed = {
            self._config.target_column: self._normalize_float(target_value),
            self._config.state_column_name: state,
            self._config.zipcode_column_name: zipcode,
            "sale_year": str(parsed_date.year),
            "sale_month": str(parsed_date.month),
            "sale_day": str(parsed_date.day),
        }

        for column in self._config.numeric_columns:
            value_raw = row.get(column, "").strip()
            try:
                transformed[column] = self._normalize_float(float(value_raw))
            except ValueError as exc:
                raise DataTransformationError(
                    f"Numeric column `{column}` contains a non-numeric value at line {index}: `{value_raw}`."
                ) from exc

        for column in self._config.categorical_columns:
            transformed[column] = row.get(column, "").strip()

        for column in self._config.drop_columns:
            transformed.pop(column, None)

        transformed.pop(self._config.statezip_column, None)
        transformed.pop(self._config.date_column, None)

        return transformed

    def _write_transformed_csv(
        self,
        output_columns: tuple[str, ...],
        rows: list[dict[str, str]],
    ) -> None:
        with self._config.transformed_data_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=list(output_columns))
            writer.writeheader()
            writer.writerows(rows)

    def _write_metadata(
        self,
        *,
        input_row_count: int,
        output_row_count: int,
        dropped_row_count: int,
        output_columns: tuple[str, ...],
    ) -> None:
        metadata = {
            "generated_at": datetime.now(UTC).isoformat(),
            "input_data_path": str(self._config.input_data_path),
            "transformed_data_path": str(self._config.transformed_data_path),
            "input_row_count": input_row_count,
            "output_row_count": output_row_count,
            "dropped_row_count": dropped_row_count,
            "target_column": self._config.target_column,
            "feature_columns": [
                column for column in output_columns if column != self._config.target_column
            ],
            "drop_columns": list(self._config.drop_columns),
            "derived_columns": [
                self._config.state_column_name,
                self._config.zipcode_column_name,
                "sale_year",
                "sale_month",
                "sale_day",
            ],
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

    @staticmethod
    def _split_statezip(value: str) -> tuple[str, str]:
        parts = value.split(maxsplit=1)
        if len(parts) == 2:
            return parts[0], parts[1]
        if len(parts) == 1:
            return parts[0], ""
        return "", ""

    @staticmethod
    def _normalize_float(value: float) -> str:
        if value.is_integer():
            return str(int(value))
        return f"{value:.6f}".rstrip("0").rstrip(".")
