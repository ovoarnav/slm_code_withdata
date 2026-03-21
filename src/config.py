from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TECH_NOTES_PATH = DATA_DIR / "technician_notes.csv"

MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LENGTH = 128
RANDOM_STATE = 42

SMOKE_TEST = False
SMOKE_TEST_ROWS = 1000
