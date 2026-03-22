from __future__ import annotations

from pathlib import Path

import pandas as pd

from inference import predict_failure
from raw import run_salvage_engine, save_outputs, load_csv, build_example_data


# =========================================================
# PATH CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).expanduser().resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "dockmaster_synthetic_dataset_csv"
OUTPUT_DIR = PROJECT_ROOT / "salvage_engine_outputs"

VESSELS_PATH = DATA_DIR / "vessels.csv"
SLM_PART_MAPPING_PATH = DATA_DIR / "slm_part_mapping.csv"
VESSEL_PARTS_CATALOG_PATH = DATA_DIR / "vessel_parts_catalog.csv"
PART_FAILURE_RULES_PATH = DATA_DIR / "part_failure_rules.csv"
PARTS_MARKET_VALUES_PATH = DATA_DIR / "parts_market_values.csv"


# =========================================================
# HELPERS
# =========================================================
def _assert_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def _load_real_inputs() -> dict[str, pd.DataFrame]:
    _assert_file_exists(VESSELS_PATH, "vessels.csv")
    _assert_file_exists(SLM_PART_MAPPING_PATH, "slm_part_mapping.csv")
    _assert_file_exists(VESSEL_PARTS_CATALOG_PATH, "vessel_parts_catalog.csv")
    _assert_file_exists(PART_FAILURE_RULES_PATH, "part_failure_rules.csv")
    _assert_file_exists(PARTS_MARKET_VALUES_PATH, "parts_market_values.csv")

    return {
        "vessels_df": load_csv(VESSELS_PATH),
        "slm_part_mapping_df": load_csv(SLM_PART_MAPPING_PATH),
        "vessel_parts_catalog_df": load_csv(VESSEL_PARTS_CATALOG_PATH),
        "part_failure_rules_df": load_csv(PART_FAILURE_RULES_PATH),
        "parts_market_values_df": load_csv(PARTS_MARKET_VALUES_PATH),
    }


def _load_inputs_with_fallback() -> dict[str, pd.DataFrame]:
    try:
        print("\n=== LOADING REAL CSV INPUTS ===")
        print(f"DATA_DIR: {DATA_DIR}")
        return _load_real_inputs()
    except FileNotFoundError as e:
        print(f"\n[WARN] {e}")
        print("Falling back to build_example_data() from raw.py")
        return build_example_data()


def build_slm_outputs_df(
    vessel_id: str,
    note_text: str,
    note_id: str,
) -> pd.DataFrame:
    prediction = predict_failure(note_text)

    print("\n=== INFERENCE OUTPUT ===")
    print(prediction)

    # IMPORTANT:
    # Do NOT include "subsystem" here because raw.py merges on slm_label
    # with slm_part_mapping_df, which already has a subsystem column.
    # If both sides contain "subsystem", pandas creates subsystem_x/subsystem_y
    # and raw.py then crashes looking for plain "subsystem".
    return pd.DataFrame(
        [
            {
                "vessel_id": str(vessel_id),
                "slm_label": str(prediction["failure_mode"]),
                "confidence": float(prediction["failure_mode_confidence"]),
                "note_id": str(note_id),
            }
        ]
    )


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline_from_prompt() -> None:
    print("=== SALVAGE PIPELINE ===")

    data = _load_inputs_with_fallback()
    vessels_df = data["vessels_df"].copy()
    vessels_df["vessel_id"] = vessels_df["vessel_id"].astype(str).str.strip()

    available_vessel_ids = sorted(vessels_df["vessel_id"].dropna().unique().tolist())
    print(f"\nAvailable vessel_ids: {available_vessel_ids[:25]}")
    if len(available_vessel_ids) > 25:
        print(f"... and {len(available_vessel_ids) - 25} more")

    vessel_id = input("\nEnter vessel_id: ").strip()
    if not vessel_id:
        raise ValueError("vessel_id cannot be empty.")

    if vessel_id not in set(available_vessel_ids):
        raise ValueError(
            f"vessel_id '{vessel_id}' not found. "
            f"Available examples: {available_vessel_ids[:25]}"
        )

    note_id = input("Enter note_id (press enter for default): ").strip() or "note_001"

    note_text = input("Enter technician note: ").strip()
    if not note_text:
        raise ValueError("Technician note cannot be empty.")

    slm_outputs_df = build_slm_outputs_df(
        vessel_id=vessel_id,
        note_text=note_text,
        note_id=note_id,
    )

    print("\n=== SLM OUTPUTS DF ===")
    print(slm_outputs_df)

    result = run_salvage_engine(
        vessels_df=data["vessels_df"],
        slm_outputs_df=slm_outputs_df,
        slm_part_mapping_df=data["slm_part_mapping_df"],
        vessel_parts_catalog_df=data["vessel_parts_catalog_df"],
        part_failure_rules_df=data["part_failure_rules_df"],
        parts_market_values_df=data["parts_market_values_df"],
    )

    print("\n=== MAPPED FAILURES ===")
    print(result.mapped_failures_df)

    print("\n=== SCORED PARTS ===")
    scored_cols = [
        "vessel_id",
        "part_id",
        "part_name",
        "subsystem",
        "status",
        "discount_pct",
        "gross_part_value",
        "adjusted_part_value",
        "rule_applied",
        "reason",
    ]
    existing_scored_cols = [c for c in scored_cols if c in result.scored_parts_df.columns]
    print(result.scored_parts_df[existing_scored_cols])

    print("\n=== VESSEL SUMMARY ===")
    print(result.vessel_summary_df)

    save_outputs(result, OUTPUT_DIR)
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


# =========================================================
# RUN
# =========================================================
if __name__ == "__main__":
    run_pipeline_from_prompt()