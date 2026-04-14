# Housing Price ML Service

## Overview

This repository is a production-ready model pipeline for predicting house prices. It is designed to support training, evaluation, and deployment of a regression model that estimates the market value of residential properties.

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

1. Add or ingest the housing dataset into `data/`.
2. Define the dataset schema and model settings in `config/config.yaml`.
3. Run the training pipeline in `src/`.
4. Evaluate model quality using `MAE` and `RMSE`.
5. Deploy the trained model from `deployment/`.

## Recommended First Step

1. Collect a house price dataset.
2. Confirm the target is `price` or `sale_price`.
3. Choose meaningful input features such as:
   - neighborhood, location, lot size
   - square footage, bedrooms, bathrooms
   - year built, condition, garage size
4. Start with a small prototype in `notebooks/` or `src/`.

## Notes

- Keep `uv.toml` and `uv.lock` as the source of truth for dependency management.
- Use `MAE` as the primary production metric, with `RMSE` for additional model selection insight.

---

_Last updated: April 2026_