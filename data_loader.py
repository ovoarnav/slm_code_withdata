from pathlib import Path
from typing import cast

import pandas as pd

REQUIRED_COLS = [
    "note_text",
    "extracted_subsystem",
    "extracted_failure_mode",
]


def load_technician_notes(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {csv_path}")

    csv_file = cast(str, str(csv_path))
    df = pd.read_csv(cast(str, str(csv_path)), encoding="utf-8")  # type: ignore[arg-type]

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=REQUIRED_COLS).copy()
    df["note_text"] = df["note_text"].astype(str).str.strip()
    df = df[df["note_text"] != ""].copy()

    return df.reset_index(drop=True)