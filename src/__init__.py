"""Core package for the housing data pipeline."""

from src.configuration import DataIngestionConfig, load_data_ingestion_config
from src.data_ingestion import DataIngestionService, IngestionResult

__all__ = [
    "DataIngestionConfig",
    "DataIngestionService",
    "IngestionResult",
    "load_data_ingestion_config",
]
