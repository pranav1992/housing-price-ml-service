# House Price Prediction Service

## Overview

This repository implements an end-to-end tabular ML pipeline for housing price prediction:

- data ingestion from Hugging Face
- schema and data-quality validation
- deterministic transformation into a model-ready CSV
- deterministic train/test split persistence
- model training
- model evaluation
- batch prediction / inference

The target in the current dataset is `price`. The current default model is `linear_regression`, and the trained artifact persists the full preprocessing-and-model pipeline in one pickle file.

## Current Pipeline

Running `main.py` executes these stages in order:

1. `DataIngestionService`
2. `DataValidationService`
3. `DataTransformationService`
4. `DataSplitService`
5. `ModelTrainingService`
6. `ModelEvaluationService`
7. `ModelInferenceService`

Current behavior:

- downloads `USA Housing Dataset.csv` from Hugging Face if it is not already available
- reuses the local file if it already exists
- computes and prints the dataset SHA-256 hash
- validates schema, nulls, duplicates, numeric fields, datetime parsing, and basic range rules
- emits a warning for non-positive `price` values instead of failing the pipeline
- drops rows with `price <= 0` during transformation
- writes a transformed dataset and transformation metadata under `artifacts/processed/`
- writes deterministic train/test split files and split metadata under `artifacts/splits/`
- trains the configured model and persists the fitted pipeline artifact under `artifacts/models/`
- evaluates the model on the held-out split and writes metrics plus sample predictions under `artifacts/evaluation/`
- runs batch inference on the configured input dataset and writes prediction outputs under `artifacts/predictions/`

## Repository Structure

- `main.py` — top-level pipeline entrypoint
- `config/config.yaml` — ingestion source and raw-data paths
- `config/model_config.yaml` — validation, transformation, split, training, evaluation, and inference settings
- `src/configuration.py` — config loading and validation
- `src/data_ingestion.py` — ingestion service
- `src/data_validation.py` — validation service
- `src/data_transformation.py` — transformation service
- `src/data_split.py` — deterministic split service
- `src/model_training.py` — training service and supported model options
- `src/model_evaluation.py` — evaluation service and metric generation
- `src/model_inference.py` — batch prediction service
- `src/exceptions.py` — project exception types
- `notebooks/usa_housing_eda.ipynb` — exploratory data analysis notebook
- `notebooks/feature_and_model_selection_analysis.ipynb` — technical model-selection review notebook
- `notebooks/stakeholder_project_review.ipynb` — stakeholder-facing summary notebook
- `test/` — unit tests for ingestion, validation, transformation, and main orchestration
- `data/` — raw downloaded data
- `artifacts/` — processed data, split data, model, evaluation, and prediction artifacts

## Runtime Requirements

- Python `>=3.12`
- Runtime dependency:
  - `scikit-learn>=1.5.0`

For development and tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

For notebook work:

```bash
python -m pip install -e '.[notebooks]'
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
Data split completed: 3272 train rows and 819 test rows written to .../artifacts/splits/usa_housing_split.metadata.json
Model training completed: linear_regression fitted on 3272 rows and saved to .../artifacts/models/linear_regression_model.pkl
Model evaluation completed: MAE=129055.97, RMSE=239475.45, R2=0.6389
Model inference completed: 819 rows written to .../artifacts/predictions/linear_regression_predictions.csv
```

## Input And Output Paths

Raw data inputs:

- source config: `config/config.yaml`
- downloaded cache file: `data/external/USA Housing Dataset.csv`
- materialized raw dataset: `data/raw/usa-housing-dataset/USA Housing Dataset.csv`
- ingestion manifest: `data/external/usa-housing-dataset.manifest.json`

Processed outputs:

- transformed dataset: `artifacts/processed/usa_housing_transformed.csv`
- transformation metadata: `artifacts/processed/usa_housing_transformed.metadata.json`
- train split: `artifacts/splits/usa_housing_train.csv`
- test split: `artifacts/splits/usa_housing_test.csv`
- split metadata: `artifacts/splits/usa_housing_split.metadata.json`
- model artifact: `artifacts/models/linear_regression_model.pkl`
- model metadata: `artifacts/models/linear_regression_model.metadata.json`
- evaluation metrics: `artifacts/evaluation/linear_regression_metrics.json`
- evaluation metadata: `artifacts/evaluation/linear_regression_evaluation.metadata.json`
- evaluation sample predictions: `artifacts/evaluation/linear_regression_sample_predictions.csv`
- batch inference predictions: `artifacts/predictions/linear_regression_predictions.csv`
- batch inference metadata: `artifacts/predictions/linear_regression_predictions.metadata.json`

## Configuration

`config/config.yaml` controls ingestion:

- `source_name`
- `source_url`
- `file_name`
- download/cache directories
- manifest path

`config/model_config.yaml` controls the rest of the pipeline:

- validation rules
- transformation input and output paths
- split input and output paths
- training input, model, and artifact paths
- evaluation input and output paths
- inference input and output paths
- test size and random seed for deterministic split/evaluation behavior
- required columns
- numeric and datetime columns
- strict-positive and warning-only checks
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

## Split, Training, Evaluation, And Inference

The current split stage:

- reads the transformed dataset
- writes deterministic train and test CSV files using the configured `test_size` and `random_state`
- persists split metadata so downstream stages do not recompute the split independently

The current training stage:

- reads the persisted train split
- supports `linear_regression`, `random_forest`, and `hist_gradient_boosting`
- persists the fitted scikit-learn pipeline artifact and model metadata

The current evaluation stage:

- reads the persisted test split
- loads the trained model artifact
- computes `MAE`, `RMSE`, and `R2`
- writes evaluation metrics, evaluation metadata, and sample predictions

The current inference stage:

- loads the trained model artifact
- predicts on the configured batch input dataset
- writes a prediction CSV and inference metadata
- preserves the original target as `actual_price` in the output when the input already includes `price`

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
- split behavior
- training behavior
- evaluation behavior
- inference behavior
- main pipeline orchestration

## Current Dataset Notes

- Current source: `Nhule0502/USA_house_price` on Hugging Face
- Current dataset row count before transformation: `4140`
- Current transformed row count after dropping non-positive targets: `4091`
- Current train row count: `3272`
- Current test row count: `819`
- Current known issue in source data: `49` rows have non-positive `price`

## Model Status

- Current default model: `linear_regression`
- Current held-out metrics:
  - `MAE=129055.97`
  - `RMSE=239475.45`
  - `R2=0.6389`
- Challenger results on the same held-out split:
  - `random_forest` improved MAE slightly but was worse on RMSE and R2
  - `hist_gradient_boosting` underperformed the baseline on MAE, RMSE, and R2

## Future Work

- residual analysis by city and price band
- stronger challenger models and hyperparameter tuning
- API or CLI serving layer for single-record prediction requests
- model selection automation and comparison reporting
- production-hardening around schema versioning and deployment

---

_Last updated: April 2026_
