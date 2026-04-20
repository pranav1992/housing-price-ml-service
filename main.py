from pathlib import Path

from src.configuration import load_data_ingestion_config
from src.data_ingestion import DataIngestionService


def main() -> None:
    project_root = Path(__file__).resolve().parent
    config = load_data_ingestion_config(project_root / "config" / "config.yaml", project_root=project_root)
    result = DataIngestionService(config).run()

    if result.status == "already_available":
        print(f"Data is already available with hash: {result.sha256}")
        return

    print(f"Data downloaded successfully with hash: {result.sha256}")


if __name__ == "__main__":
    main()
