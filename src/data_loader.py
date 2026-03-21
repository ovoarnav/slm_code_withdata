from __future__ import annotations

import pandas as pd
from pathlib import Path


REQUIRED_COLS = [
    "note_text",
    "extracted_subsystem",
    "extracted_failure_mode",
]


def load_technician_notes(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["note_text"] = df["note_text"].astype(str).str.strip()
    df = df[df["note_text"] != ""].copy()

    return df
