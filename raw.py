from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import json
import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================
UNKNOWN_PART_ID = "unknown_part"
STATUS_SALVAGEABLE = "salvageable"
STATUS_DISCOUNTED = "discounted"
STATUS_EXCLUDED = "excluded"
STATUS_INSPECT = "inspect"


# =========================================================
# CSV COMPAT LAYER
# =========================================================
def read_csv_compat(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """
    Wrapper around pandas.read_csv that avoids the type-checker / overload issue
    and gives clean path validation.
    """
    csv_path = Path(path).expanduser().resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing file: {csv_path}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"Path is not a file: {csv_path}")

    csv_file = cast(str, str(csv_path))
    result: Any = pd.read_csv(filepath_or_buffer=csv_file, **kwargs)
    return cast(pd.DataFrame, result)


def write_csv_compat(df: pd.DataFrame, path: str | Path, **kwargs: Any) -> None:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_file = cast(str, str(output_path))
    df.to_csv(path_or_buf=csv_file, index=False, **kwargs)


def load_csv(path: str | Path) -> pd.DataFrame:
    return read_csv_compat(path, dtype=str, low_memory=False)


# =========================================================
# VALIDATION
# =========================================================
def _require_columns(df: pd.DataFrame, required: set[str], df_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{df_name} missing required columns: {sorted(missing)}")


def validate_inputs(
    vessels_df: pd.DataFrame,
    slm_outputs_df: pd.DataFrame,
    slm_part_mapping_df: pd.DataFrame,
    vessel_parts_catalog_df: pd.DataFrame,
    part_failure_rules_df: pd.DataFrame,
    parts_market_values_df: pd.DataFrame,
) -> None:
    _require_columns(vessels_df, {"vessel_id", "vessel_model_key"}, "vessels_df")
    _require_columns(slm_outputs_df, {"vessel_id", "slm_label"}, "slm_outputs_df")
    _require_columns(
        slm_part_mapping_df,
        {"slm_label", "part_id", "part_name", "subsystem"},
        "slm_part_mapping_df",
    )
    _require_columns(
        vessel_parts_catalog_df,
        {"vessel_model_key", "part_id", "part_name", "subsystem", "quantity", "base_used_value"},
        "vessel_parts_catalog_df",
    )
    _require_columns(
        part_failure_rules_df,
        {"failed_part_id", "affected_part_id", "effect_type"},
        "part_failure_rules_df",
    )
    _require_columns(parts_market_values_df, {"part_id"}, "parts_market_values_df")


# =========================================================
# NORMALIZATION HELPERS
# =========================================================
def normalize_vessels(vessels_df: pd.DataFrame) -> pd.DataFrame:
    df = vessels_df.copy()
    df["vessel_id"] = df["vessel_id"].astype(str).str.strip()
    df["vessel_model_key"] = df["vessel_model_key"].astype(str).str.strip()
    return df


def normalize_slm_outputs(slm_outputs_df: pd.DataFrame) -> pd.DataFrame:
    df = slm_outputs_df.copy()
    df["vessel_id"] = df["vessel_id"].astype(str).str.strip()
    df["slm_label"] = df["slm_label"].astype(str).str.strip()

    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    if "note_id" not in df.columns:
        df["note_id"] = None

    return df


def normalize_slm_mapping(slm_part_mapping_df: pd.DataFrame) -> pd.DataFrame:
    df = slm_part_mapping_df.copy()
    df["slm_label"] = df["slm_label"].astype(str).str.strip()
    df["part_id"] = df["part_id"].astype(str).str.strip()
    df["part_name"] = df["part_name"].astype(str).str.strip()
    df["subsystem"] = df["subsystem"].astype(str).str.strip()
    return df


def normalize_vessel_parts_catalog(vessel_parts_catalog_df: pd.DataFrame) -> pd.DataFrame:
    df = vessel_parts_catalog_df.copy()

    df["vessel_model_key"] = df["vessel_model_key"].astype(str).str.strip()
    df["part_id"] = df["part_id"].astype(str).str.strip()
    df["part_name"] = df["part_name"].astype(str).str.strip()
    df["subsystem"] = df["subsystem"].astype(str).str.strip()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(1).clip(lower=0)
    df["base_used_value"] = pd.to_numeric(df["base_used_value"], errors="coerce").fillna(0.0)

    if "salvageable_by_default" not in df.columns:
        df["salvageable_by_default"] = True
    else:
        df["salvageable_by_default"] = (
            df["salvageable_by_default"]
            .fillna(True)
            .astype(str)
            .str.lower()
            .isin(["true", "1", "yes", "y"])
        )

    return df


def normalize_failure_rules(part_failure_rules_df: pd.DataFrame) -> pd.DataFrame:
    df = part_failure_rules_df.copy()

    df["failed_part_id"] = df["failed_part_id"].astype(str).str.strip()
    df["affected_part_id"] = df["affected_part_id"].astype(str).str.strip()
    df["effect_type"] = df["effect_type"].astype(str).str.strip().str.lower()

    if "discount_pct" not in df.columns:
        df["discount_pct"] = 0.0
    else:
        df["discount_pct"] = pd.to_numeric(df["discount_pct"], errors="coerce").fillna(0.0)

    df["discount_pct"] = df["discount_pct"].clip(lower=0.0, upper=1.0)

    if "note" not in df.columns:
        df["note"] = None

    if "rule_confidence" not in df.columns:
        df["rule_confidence"] = np.nan
    else:
        df["rule_confidence"] = pd.to_numeric(df["rule_confidence"], errors="coerce")

    valid_effects = {"remove", "discount", "inspect"}
    invalid = sorted(set(df["effect_type"]) - valid_effects)
    if invalid:
        raise ValueError(f"Invalid effect_type values: {invalid}")

    return df


def normalize_market_values(parts_market_values_df: pd.DataFrame) -> pd.DataFrame:
    df = parts_market_values_df.copy()
    df["part_id"] = df["part_id"].astype(str).str.strip()

    for col in [
        "used_value_low",
        "used_value_mid",
        "used_value_high",
        "sell_probability",
        "days_to_sell_estimate",
    ]:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["sell_probability"] = pd.to_numeric(df["sell_probability"], errors="coerce").clip(0.0, 1.0)

    return df


# =========================================================
# STEP 1: MAP SLM OUTPUTS TO CANONICAL PARTS
# =========================================================
def map_slm_outputs_to_parts(
    slm_outputs_df: pd.DataFrame,
    slm_part_mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    outputs = normalize_slm_outputs(slm_outputs_df)
    mapping = normalize_slm_mapping(slm_part_mapping_df)

    mapped = outputs.merge(mapping, on="slm_label", how="left", suffixes=("", "_mapping"))

    if "part_id" not in mapped.columns and "part_id_mapping" in mapped.columns:
        mapped["part_id"] = mapped["part_id_mapping"]

    if "part_name" not in mapped.columns and "part_name_mapping" in mapped.columns:
        mapped["part_name"] = mapped["part_name_mapping"]

    if "subsystem" not in mapped.columns and "subsystem_mapping" in mapped.columns:
        mapped["subsystem"] = mapped["subsystem_mapping"]

    if "part_id" not in mapped.columns:
        mapped["part_id"] = UNKNOWN_PART_ID
    else:
        mapped["part_id"] = mapped["part_id"].fillna(UNKNOWN_PART_ID)

    if "part_name" not in mapped.columns:
        mapped["part_name"] = "unknown_part"
    else:
        mapped["part_name"] = mapped["part_name"].fillna("unknown_part")

    if "subsystem" not in mapped.columns:
        mapped["subsystem"] = "unknown_subsystem"
    else:
        mapped["subsystem"] = mapped["subsystem"].fillna("unknown_subsystem")

    mapped["mapping_found"] = mapped["part_id"] != UNKNOWN_PART_ID

    drop_cols = [c for c in ["part_id_mapping", "part_name_mapping", "subsystem_mapping"] if c in mapped.columns]
    if drop_cols:
        mapped = mapped.drop(columns=drop_cols)

    return mapped


# =========================================================
# STEP 2: BUILD VESSEL PARTS LIST
# =========================================================
def build_vessel_parts_list(
    vessels_df: pd.DataFrame,
    vessel_parts_catalog_df: pd.DataFrame,
) -> pd.DataFrame:
    vessels = normalize_vessels(vessels_df)
    catalog = normalize_vessel_parts_catalog(vessel_parts_catalog_df)

    parts = vessels.merge(catalog, on="vessel_model_key", how="left")

    missing_catalog = parts["part_id"].isna()
    if missing_catalog.any():
        missing_keys = sorted(parts.loc[missing_catalog, "vessel_model_key"].dropna().unique().tolist())
        print(f"[WARN] Missing catalog rows for vessel_model_keys: {missing_keys}")

    parts = parts.dropna(subset=["part_id"]).copy()

    parts["status"] = np.where(
        parts["salvageable_by_default"].fillna(True),
        STATUS_SALVAGEABLE,
        STATUS_EXCLUDED,
    )
    parts["discount_pct"] = np.where(parts["status"] == STATUS_EXCLUDED, 1.0, 0.0)
    parts["direct_failure_flag"] = False
    parts["rule_applied"] = None
    parts["reason"] = None

    return parts


# =========================================================
# STEP 3: APPLY DIRECT FAILURES AND RULES
# =========================================================
def apply_failure_rules(
    vessel_parts_df: pd.DataFrame,
    failed_parts_df: pd.DataFrame,
    part_failure_rules_df: pd.DataFrame,
) -> pd.DataFrame:
    parts = vessel_parts_df.copy()
    rules = normalize_failure_rules(part_failure_rules_df)

    required_failed_cols = {"vessel_id", "part_id", "part_name", "subsystem"}
    _require_columns(failed_parts_df, required_failed_cols, "failed_parts_df")

    failed = failed_parts_df.copy()
    failed["vessel_id"] = failed["vessel_id"].astype(str).str.strip()
    failed["part_id"] = failed["part_id"].astype(str).str.strip()

    failed_unique = failed[["vessel_id", "part_id", "part_name", "subsystem"]].drop_duplicates()

    # direct failures
    direct_hits = parts.merge(
        failed_unique[["vessel_id", "part_id"]].assign(_failed_hit=True),
        on=["vessel_id", "part_id"],
        how="left",
    )

    direct_mask = direct_hits["_failed_hit"].fillna(False).astype(bool)
    parts.loc[direct_mask, "status"] = STATUS_EXCLUDED
    parts.loc[direct_mask, "discount_pct"] = 1.0
    parts.loc[direct_mask, "direct_failure_flag"] = True
    parts.loc[direct_mask, "rule_applied"] = "direct_failure"
    parts.loc[direct_mask, "reason"] = "direct_failure"

    # downstream rule effects
    for vessel_id, vessel_failed_group in failed_unique.groupby("vessel_id", dropna=False):
        failed_ids = vessel_failed_group["part_id"].dropna().astype(str).unique().tolist()
        if not failed_ids:
            continue

        vessel_mask = parts["vessel_id"] == vessel_id
        impacted_rules = rules[rules["failed_part_id"].isin(failed_ids)].copy()

        if impacted_rules.empty:
            continue

        for _, rule in impacted_rules.iterrows():
            affected_mask = vessel_mask & (parts["part_id"] == rule["affected_part_id"])

            if not affected_mask.any():
                continue

            if rule["effect_type"] == "remove":
                parts.loc[affected_mask, "status"] = STATUS_EXCLUDED
                parts.loc[affected_mask, "discount_pct"] = 1.0
                parts.loc[affected_mask, "rule_applied"] = f"remove_from_{rule['failed_part_id']}"
                parts.loc[affected_mask, "reason"] = rule["note"]

            elif rule["effect_type"] == "discount":
                current = parts.loc[affected_mask, "discount_pct"].fillna(0.0)
                updated = np.maximum(current, float(rule["discount_pct"]))
                parts.loc[affected_mask, "discount_pct"] = updated

                not_excluded = affected_mask & (parts["status"] != STATUS_EXCLUDED)
                parts.loc[not_excluded, "status"] = STATUS_DISCOUNTED
                parts.loc[affected_mask, "rule_applied"] = f"discount_from_{rule['failed_part_id']}"
                parts.loc[affected_mask, "reason"] = rule["note"]

            elif rule["effect_type"] == "inspect":
                inspect_mask = affected_mask & (parts["status"] == STATUS_SALVAGEABLE)
                parts.loc[inspect_mask, "status"] = STATUS_INSPECT
                parts.loc[inspect_mask, "rule_applied"] = f"inspect_from_{rule['failed_part_id']}"
                parts.loc[inspect_mask, "reason"] = rule["note"]

    parts["discount_pct"] = pd.to_numeric(parts["discount_pct"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    return parts


# =========================================================
# STEP 4: SCORE REMAINING PARTS
# =========================================================
def score_remaining_parts(
    parts_with_status_df: pd.DataFrame,
    parts_market_values_df: pd.DataFrame,
) -> pd.DataFrame:
    parts = parts_with_status_df.copy()
    market = normalize_market_values(parts_market_values_df)

    scored = parts.merge(market, on="part_id", how="left")

    scored["market_reference_value"] = (
        scored["used_value_mid"]
        .fillna(scored["used_value_high"])
        .fillna(scored["used_value_low"])
    )

    scored["effective_base_value"] = scored["base_used_value"].where(
        scored["base_used_value"].notna() & (scored["base_used_value"] > 0),
        scored["market_reference_value"],
    )
    scored["effective_base_value"] = pd.to_numeric(scored["effective_base_value"], errors="coerce").fillna(0.0)
    scored["quantity"] = pd.to_numeric(scored["quantity"], errors="coerce").fillna(1)
    scored["discount_pct"] = pd.to_numeric(scored["discount_pct"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    scored["gross_part_value"] = scored["effective_base_value"] * scored["quantity"]
    scored["adjusted_part_value"] = scored["gross_part_value"] * (1.0 - scored["discount_pct"])

    excluded_mask = scored["status"] == STATUS_EXCLUDED
    scored.loc[excluded_mask, "adjusted_part_value"] = 0.0

    return scored


# =========================================================
# STEP 5: AGGREGATE TO VESSEL LEVEL
# =========================================================
def summarize_vessel_salvage(scored_parts_df: pd.DataFrame) -> pd.DataFrame:
    df = scored_parts_df.copy()

    summary = (
        df.groupby("vessel_id", dropna=False)
        .agg(
            total_parts=("part_id", "size"),
            salvageable_parts_count=("status", lambda s: int((s == STATUS_SALVAGEABLE).sum())),
            discounted_parts_count=("status", lambda s: int((s == STATUS_DISCOUNTED).sum())),
            excluded_parts_count=("status", lambda s: int((s == STATUS_EXCLUDED).sum())),
            inspect_parts_count=("status", lambda s: int((s == STATUS_INSPECT).sum())),
            gross_parts_value=("gross_part_value", "sum"),
            adjusted_salvage_value=("adjusted_part_value", "sum"),
        )
        .reset_index()
    )

    summary["salvage_retention_ratio"] = np.where(
        summary["gross_parts_value"] > 0,
        summary["adjusted_salvage_value"] / summary["gross_parts_value"],
        0.0,
    )

    return summary


# =========================================================
# RESULT OBJECT
# =========================================================
@dataclass
class SalvageEngineResult:
    mapped_failures_df: pd.DataFrame
    vessel_parts_logic_df: pd.DataFrame
    scored_parts_df: pd.DataFrame
    vessel_summary_df: pd.DataFrame


# =========================================================
# MAIN ORCHESTRATOR
# =========================================================
def run_salvage_engine(
    vessels_df: pd.DataFrame,
    slm_outputs_df: pd.DataFrame,
    slm_part_mapping_df: pd.DataFrame,
    vessel_parts_catalog_df: pd.DataFrame,
    part_failure_rules_df: pd.DataFrame,
    parts_market_values_df: pd.DataFrame,
) -> SalvageEngineResult:
    validate_inputs(
        vessels_df=vessels_df,
        slm_outputs_df=slm_outputs_df,
        slm_part_mapping_df=slm_part_mapping_df,
        vessel_parts_catalog_df=vessel_parts_catalog_df,
        part_failure_rules_df=part_failure_rules_df,
        parts_market_values_df=parts_market_values_df,
    )

    mapped_failures_df = map_slm_outputs_to_parts(
        slm_outputs_df=slm_outputs_df,
        slm_part_mapping_df=slm_part_mapping_df,
    )

    vessel_parts_df = build_vessel_parts_list(
        vessels_df=vessels_df,
        vessel_parts_catalog_df=vessel_parts_catalog_df,
    )

    vessel_parts_logic_df = apply_failure_rules(
        vessel_parts_df=vessel_parts_df,
        failed_parts_df=mapped_failures_df,
        part_failure_rules_df=part_failure_rules_df,
    )

    scored_parts_df = score_remaining_parts(
        parts_with_status_df=vessel_parts_logic_df,
        parts_market_values_df=parts_market_values_df,
    )

    vessel_summary_df = summarize_vessel_salvage(scored_parts_df)

    return SalvageEngineResult(
        mapped_failures_df=mapped_failures_df,
        vessel_parts_logic_df=vessel_parts_logic_df,
        scored_parts_df=scored_parts_df,
        vessel_summary_df=vessel_summary_df,
    )


# =========================================================
# SAVE OUTPUTS
# =========================================================
def save_outputs(result: SalvageEngineResult, output_dir: str | Path) -> None:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    write_csv_compat(result.mapped_failures_df, output_path / "mapped_failures.csv")
    write_csv_compat(result.vessel_parts_logic_df, output_path / "vessel_parts_logic.csv")
    write_csv_compat(result.scored_parts_df, output_path / "scored_parts.csv")
    write_csv_compat(result.vessel_summary_df, output_path / "vessel_summary.csv")

    metadata = {
        "n_mapped_failures": int(len(result.mapped_failures_df)),
        "n_logic_rows": int(len(result.vessel_parts_logic_df)),
        "n_scored_parts": int(len(result.scored_parts_df)),
        "n_vessels": int(result.vessel_summary_df["vessel_id"].nunique()),
    }

    metadata_file = cast(str, str((output_path / "run_metadata.json").resolve()))
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


# =========================================================
# OPTIONAL FILE-BASED RUNNER
# =========================================================
def run_salvage_engine_from_files(
    vessels_path: str | Path,
    slm_outputs_path: str | Path,
    slm_part_mapping_path: str | Path,
    vessel_parts_catalog_path: str | Path,
    part_failure_rules_path: str | Path,
    parts_market_values_path: str | Path,
    output_dir: str | Path,
) -> SalvageEngineResult:
    vessels_df = load_csv(vessels_path)
    slm_outputs_df = load_csv(slm_outputs_path)
    slm_part_mapping_df = load_csv(slm_part_mapping_path)
    vessel_parts_catalog_df = load_csv(vessel_parts_catalog_path)
    part_failure_rules_df = load_csv(part_failure_rules_path)
    parts_market_values_df = load_csv(parts_market_values_path)

    result = run_salvage_engine(
        vessels_df=vessels_df,
        slm_outputs_df=slm_outputs_df,
        slm_part_mapping_df=slm_part_mapping_df,
        vessel_parts_catalog_df=vessel_parts_catalog_df,
        part_failure_rules_df=part_failure_rules_df,
        parts_market_values_df=parts_market_values_df,
    )

    save_outputs(result, output_dir)
    return result


# =========================================================
# EXAMPLE DATA
# =========================================================
def build_example_data() -> dict[str, pd.DataFrame]:
    vessels_df = pd.DataFrame(
        {
            "vessel_id": ["101", "102"],
            "vessel_model_key": ["yamaha_242x_2020", "sea_ray_spx_190_2019"],
            "make": ["Yamaha", "Sea Ray"],
            "model": ["242X", "SPX 190"],
            "year": ["2020", "2019"],
        }
    )

    slm_outputs_df = pd.DataFrame(
        {
            "vessel_id": ["101", "101", "102"],
            "slm_label": ["raw_water_pump_failure", "electrical_panel_fault", "alternator_failure"],
            "confidence": [0.95, 0.83, 0.91],
        }
    )

    slm_part_mapping_df = pd.DataFrame(
        {
            "slm_label": ["raw_water_pump_failure", "electrical_panel_fault", "alternator_failure"],
            "part_id": ["raw_water_pump", "electrical_panel", "alternator"],
            "part_name": ["Raw Water Pump", "Electrical Panel", "Alternator"],
            "subsystem": ["cooling", "electrical", "electrical"],
        }
    )

    vessel_parts_catalog_df = pd.DataFrame(
        {
            "vessel_model_key": [
                "yamaha_242x_2020",
                "yamaha_242x_2020",
                "yamaha_242x_2020",
                "yamaha_242x_2020",
                "sea_ray_spx_190_2019",
                "sea_ray_spx_190_2019",
                "sea_ray_spx_190_2019",
            ],
            "part_id": [
                "raw_water_pump",
                "heat_exchanger",
                "electrical_panel",
                "wiring_harness",
                "alternator",
                "battery_charger",
                "wiring_harness",
            ],
            "part_name": [
                "Raw Water Pump",
                "Heat Exchanger",
                "Electrical Panel",
                "Wiring Harness",
                "Alternator",
                "Battery Charger",
                "Wiring Harness",
            ],
            "subsystem": [
                "cooling",
                "cooling",
                "electrical",
                "electrical",
                "electrical",
                "electrical",
                "electrical",
            ],
            "quantity": [1, 1, 1, 1, 1, 1, 1],
            "base_used_value": [450, 1100, 800, 300, 350, 220, 280],
        }
    )

    part_failure_rules_df = pd.DataFrame(
        {
            "failed_part_id": [
                "raw_water_pump",
                "electrical_panel",
                "electrical_panel",
                "alternator",
            ],
            "affected_part_id": [
                "heat_exchanger",
                "wiring_harness",
                "battery_charger",
                "wiring_harness",
            ],
            "effect_type": [
                "discount",
                "discount",
                "inspect",
                "inspect",
            ],
            "discount_pct": [
                0.30,
                0.40,
                0.0,
                0.0,
            ],
            "note": [
                "Cooling failure may have damaged exchanger",
                "Panel fault may affect harness",
                "Panel fault requires charger inspection",
                "Alternator failure may require harness inspection",
            ],
        }
    )

    parts_market_values_df = pd.DataFrame(
        {
            "part_id": [
                "raw_water_pump",
                "heat_exchanger",
                "electrical_panel",
                "wiring_harness",
                "alternator",
                "battery_charger",
            ],
            "used_value_low": [300, 800, 500, 150, 200, 120],
            "used_value_mid": [425, 1050, 760, 250, 320, 190],
            "used_value_high": [500, 1200, 900, 320, 400, 240],
            "sell_probability": [0.72, 0.60, 0.58, 0.45, 0.69, 0.40],
            "days_to_sell_estimate": [18, 40, 35, 55, 21, 60],
        }
    )

    return {
        "vessels_df": vessels_df,
        "slm_outputs_df": slm_outputs_df,
        "slm_part_mapping_df": slm_part_mapping_df,
        "vessel_parts_catalog_df": vessel_parts_catalog_df,
        "part_failure_rules_df": part_failure_rules_df,
        "parts_market_values_df": parts_market_values_df,
    }


# =========================================================
# RUN BLOCK
# =========================================================
if __name__ == "__main__":
    # OPTION A: example run
    data = build_example_data()

    result = run_salvage_engine(
        vessels_df=data["vessels_df"],
        slm_outputs_df=data["slm_outputs_df"],
        slm_part_mapping_df=data["slm_part_mapping_df"],
        vessel_parts_catalog_df=data["vessel_parts_catalog_df"],
        part_failure_rules_df=data["part_failure_rules_df"],
        parts_market_values_df=data["parts_market_values_df"],
    )

    print("\n=== MAPPED FAILURES ===")
    print(result.mapped_failures_df)

    print("\n=== SCORED PARTS ===")
    print(
        result.scored_parts_df[
            [
                "vessel_id",
                "part_id",
                "part_name",
                "status",
                "discount_pct",
                "gross_part_value",
                "adjusted_part_value",
                "rule_applied",
                "reason",
            ]
        ]
    )

    print("\n=== VESSEL SUMMARY ===")
    print(result.vessel_summary_df)

    save_outputs(result, "salvage_engine_outputs")

    # OPTION B: real-file run
    # Uncomment and replace with your actual files
    #
    # result = run_salvage_engine_from_files(
    #     vessels_path="/Users/arnav/PycharmProjects/PythonProject1/vessels.csv",
    #     slm_outputs_path="/Users/arnav/PycharmProjects/PythonProject1/slm_outputs.csv",
    #     slm_part_mapping_path="/Users/arnav/PycharmProjects/PythonProject1/slm_part_mapping.csv",
    #     vessel_parts_catalog_path="/Users/arnav/PycharmProjects/PythonProject1/vessel_parts_catalog.csv",
    #     part_failure_rules_path="/Users/arnav/PycharmProjects/PythonProject1/part_failure_rules.csv",
    #     parts_market_values_path="/Users/arnav/PycharmProjects/PythonProject1/parts_market_values.csv",
    #     output_dir="/Users/arnav/PycharmProjects/PythonProject1/salvage_engine_outputs",
    # )