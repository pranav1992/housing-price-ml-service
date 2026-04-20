# House Rent Prediction Service

## Overview

This repository currently implements the first three stages of a tabular ML pipeline for housing price prediction:

- data ingestion from Hugging Face
- schema and data-quality validation
- deterministic transformation into a model-ready CSV

The target in the current dataset is `price`. The project is not yet training a model, but the raw data pipeline is now in place and tested.

## Current Pipeline

Running `main.py` executes these stages in order:

1. `DataIngestionService`
2. `DataValidationService`
3. `DataTransformationService`

Current behavior:

- downloads `USA Housing Dataset.csv` from Hugging Face if it is not already available
- reuses the local file if it already exists
- computes and prints the dataset SHA-256 hash
- validates schema, nulls, duplicates, numeric fields, datetime parsing, and basic range rules
- emits a warning for non-positive `price` values instead of failing the pipeline
- drops rows with `price <= 0` during transformation
- writes a transformed dataset and transformation metadata under `artifacts/processed/`

## Repository Structure

- `main.py` — top-level pipeline entrypoint
- `config/config.yaml` — ingestion source and raw-data paths
- `config/model_config.yaml` — validation and transformation settings
- `src/configuration.py` — config loading and validation
- `src/data_ingestion.py` — ingestion service
- `src/data_validation.py` — validation service
- `src/data_transformation.py` — transformation service
- `src/exceptions.py` — project exception types
- `notebooks/usa_housing_eda.ipynb` — exploratory data analysis notebook
- `test/` — unit tests for ingestion, validation, transformation, and main orchestration
- `data/` — raw downloaded data
- `artifacts/` — processed outputs and future model artifacts

## Runtime Requirements

- Python `>=3.12`
- No third-party runtime dependencies are required for the pipeline code itself

For development and tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

## Run The Pipeline

From the repo root:

```bash
python3 main.py
```

Typical output:

```text
Data is already available with hash: <sha256>
Data validation passed with warnings: Column `price` contains 49 non-positive value(s).
Data transformation completed: 4091 rows written to .../artifacts/processed/usa_housing_transformed.csv
```

## Input And Output Paths

Raw data inputs:

- source config: `config/config.yaml`
- downloaded cache file: `data/raw/USA Housing Dataset.csv`
- materialized raw dataset: `data/raw/usa-housing-dataset/USA Housing Dataset.csv`
- ingestion manifest: `data/raw/usa-housing-dataset.manifest.json`

Processed outputs:

- transformed dataset: `artifacts/processed/usa_housing_transformed.csv`
- transformation metadata: `artifacts/processed/usa_housing_transformed.metadata.json`

## Configuration

`config/config.yaml` controls ingestion:

- `source_name`
- `source_url`
- `file_name`
- raw data directories
- manifest path

`config/model_config.yaml` controls validation and transformation:

- required columns
- numeric and datetime columns
- strict-positive and warning-only checks
- transformation input and output paths
- target column
- derived features from `date`
- `statezip` splitting into `state` and `zipcode`
- row-dropping rule for non-positive target values

## Validation Rules

The current validation stage enforces:

- expected schema for the downloaded CSV
- non-empty required fields
- numeric parsing for configured numeric columns
- `%Y-%m-%d %H:%M:%S` parsing for `date`
- no exact duplicate rows
- strict-positive checks for structural columns such as `sqft_living`, `sqft_lot`, `sqft_above`, and `yr_built`

The current validation stage warns on:

- non-positive `price`
- unexpected values in columns configured as constants, currently `country=USA`

## Transformation Rules

The current transformation stage:

- drops rows where `price <= 0`
- keeps the configured numeric features
- keeps `city` as the current categorical feature
- drops `street`, `country`, raw `statezip`, and raw `date`
- derives:
  - `state`
  - `zipcode`
  - `sale_year`
  - `sale_month`
  - `sale_day`

## EDA Notebook

Notebook:

- `notebooks/usa_housing_eda.ipynb`

Purpose:

- inspect schema and row counts
- review target distribution and outliers
- profile categorical and temporal fields
- turn exploratory findings into validation and transformation decisions

Important kernel note:

- The notebook works best with a local Python kernel that can access this repo on disk.
- If you use a Google Colab kernel from VS Code, that remote kernel usually cannot see your local `data/` directory unless you upload or clone the repo into the Colab runtime.

The notebook includes path-resolution debug output to make that failure mode obvious.

## Tests

Run the full test suite:

```bash
./.venv/bin/python -m pytest -q
```

Current automated coverage includes:

- ingestion behavior
- validation behavior
- transformation behavior
- main pipeline orchestration

## Current Dataset Notes

- Current source: `Nhule0502/USA_house_price` on Hugging Face
- Current dataset row count before transformation: `4140`
- Current transformed row count after dropping non-positive targets: `4091`
- Current known issue in source data: `49` rows have non-positive `price`

## Next Planned Work

- train/test split stage
- model training stage
- evaluation metrics and model selection
- artifact persistence for trained models and preprocessors
- prediction / inference pipeline

---

_Last updated: April 2026_
