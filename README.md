# House Rent Prediction Service

## Overview

This repository is a production-ready model pipeline for predicting house rent. It is designed to support training, evaluation, and deployment of a regression model that estimates rental prices for residential properties.

## Objective

- Predict the sale price of a house.
- Use structured housing features such as location, size, age, condition, and amenities.
- Track model performance using regression metrics like `MAE` and `RMSE`.

## Repository Structure

- `data/` — raw and processed datapaths for training and evaluation.
- `config/` — configuration files for model parameters and data schema.
- `src/` — source code for data processing, training, and evaluation.
- `notebooks/` — exploratory analysis and prototyping notebooks.
- `deployment/` — deployment artifacts and serving configuration.
- `docker/` — Docker setup for reproducible environments.
- `test/` — unit tests and validation code.
- `artifacts/` — generated model artifacts and experiment outputs.
- `scripts/` — convenience scripts for setup, training, and deployment.

## Setup

This project uses `uv` as the package manager.

```bash
uv install
```

If you want pip compatibility, export requirements from the lock file later:

```bash
uv export -r requirements.txt
```

## How to Use

1. Review the Hugging Face source settings in `config/config.yaml`.
2. Run the data ingestion pipeline:

```bash
python main.py
```

3. The pipeline will:
   - download the dataset CSV from Hugging Face into `data/raw/`
   - skip the download if the dataset file already exists locally
   - print the dataset file SHA-256 hash
   - materialize the dataset into `data/raw/usa-housing-dataset`
   - write a manifest to `data/raw/usa-housing-dataset.manifest.json`
   - validate the ingested CSV against schema and data-quality checks
4. Review the exploratory notebook in `notebooks/usa_housing_eda.ipynb`.
5. Continue with downstream processing and model training in `src/`.

## Notes

- The current dataset source is Hugging Face: `Nhule0502/USA_house_price`.
- Current validation warns that `price` contains non-positive values in the downloaded dataset.
- Use `MAE` as the primary production metric, with `RMSE` for additional model selection insight.

---

_Last updated: April 2026_
