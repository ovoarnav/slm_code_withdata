import os
import pandas as pd

REQUIRED_COLS = [
    "note_text",
    "extracted_subsystem",
    "extracted_failure_mode",
]


def load_technician_notes(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError("Missing file: {}".format(path))

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: {}".format(missing))

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["note_text"] = df["note_text"].astype(str).str.strip()
    df = df[df["note_text"] != ""].copy()

    return df
