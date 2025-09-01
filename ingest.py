from pathlib import Path
from rag_core import ingest_folder

if __name__ == "__main__":
    data_dir = Path("data")
    storage_dir = Path("storage")
    ingest_folder(data_dir, storage_dir)
