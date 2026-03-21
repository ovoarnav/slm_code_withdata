import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")

os.makedirs(ARTIFACT_DIR, exist_ok=True)

TECH_NOTES_PATH = os.path.join(DATA_DIR, "technician_notes.csv")

MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LENGTH = 128
RANDOM_STATE = 42

SMOKE_TEST = False
SMOKE_TEST_ROWS = 1000
