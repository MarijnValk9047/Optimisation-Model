from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# USER-EDITABLE CONFIG
# ---------------------------

LOCAL_TZ = "Europe/Amsterdam"

# Input from Script 1
HARMONISED_HOURLY_PARQUET = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\00_harmonised"
    r"\da_prices_harmonised_hourly.parquet"
)

# Outputs for Script 2
DAILY_FEATURES_DIR = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\01_daily_features"
)

LOGS_ROOT = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\4_logs\da_regimes"
)

# Train/val/test split
TRAIN_YEARS = (2022, 2023)
VALIDATE_YEARS = (2024,)
TEST_YEARS = (2025,)

# Season mapping (explicit modelling choice)
WINTER_MONTHS = (11, 12, 1, 2)
SUMMER_MONTHS = (6, 7, 8)
# Shoulder months are the rest: 3,4,5,9,10


# ---------------------------
# INTERNALS
# ---------------------------

@dataclass(frozen=True)
class SeasonConfig:
    winter_months: Tuple[int, ...]
    summer_months: Tuple[int, ...]


def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def season_of_month(month: int, cfg: SeasonConfig) -> str:
    if month in cfg.winter_months:
        return "winter"
    if month in cfg.summer_months:
        return "summer"
    return "shoulder"


def split_of_year(year: int) -> str:
    if year in TRAIN_YEARS:
        return "train"
    if year in VALIDATE_YEARS:
        return "validate"
    if year in TEST_YEARS:
        return "test"
    return "out_of_scope"


def daytype_of_date(dt: pd.Timestamp) -> str:
    # Monday=0 ... Sunday=6
    return "weekend" if dt.weekday() >= 5 else "weekday"


def conditioning_class(daytype: str, season: str) -> str:
    return f"{daytype}__{season}"


def build_daily_table(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Input hourly must contain:
      - ts (timezone-aware)
      - price_eur_mwh (float)
    Output is one row per local date with:
      - price_h00 .. price_h23
      - shape_z_h00 .. shape_z_h23 (only if complete day & std>0)
      - level metrics
      - class labels and split labels
      - quality flags
    """
    df = hourly.copy()

    # Ensure timezone-aware and in local tz
    if df["ts"].dt.tz is None:
        # Should not happen if Script 1 wrote parquet, but guard anyway
        df["ts"] = df["ts"].dt.tz_localize(LOCAL_TZ)
    else:
        df["ts"] = df["ts"].dt.tz_convert(LOCAL_TZ)

    # Extract local date and hour
    df["date"] = df["ts"].dt.date
    df["hour"] = df["ts"].dt.hour
    df["year"] = df["ts"].dt.year
    df["month"] = df["ts"].dt.month

    # Pivot to wide 24-hour vector
    pivot = (
        df.pivot_table(index="date", columns="hour", values="price_eur_mwh", aggfunc="mean")
        .reindex(columns=list(range(24)))
    )

    # Rename columns to price_hXX
    pivot.columns = [f"price_h{h:02d}" for h in pivot.columns]
    pivot = pivot.reset_index()

    # Compute quality flags
    price_cols = [f"price_h{h:02d}" for h in range(24)]
    pivot["missing_hours"] = pivot[price_cols].isna().sum(axis=1).astype(int)
    pivot["is_complete_24h"] = pivot["missing_hours"].eq(0)

    # Level metrics based on available hours
    pivot["level_mean"] = pivot[price_cols].mean(axis=1, skipna=True)
    pivot["level_std"] = pivot[price_cols].std(axis=1, ddof=0, skipna=True)
    pivot["level_min"] = pivot[price_cols].min(axis=1, skipna=True)
    pivot["level_max"] = pivot[price_cols].max(axis=1, skipna=True)
    pivot["level_p05"] = pivot[price_cols].quantile(0.05, axis=1, interpolation="linear")
    pivot["level_p50"] = pivot[price_cols].quantile(0.50, axis=1, interpolation="linear")
    pivot["level_p95"] = pivot[price_cols].quantile(0.95, axis=1, interpolation="linear")

    # Z-normalised shape: (price - daily_mean) / daily_std
    # Only meaningful when complete and std > 0
    shape_cols = [f"shape_z_h{h:02d}" for h in range(24)]
    pivot[shape_cols] = np.nan

    mask = pivot["is_complete_24h"] & (pivot["level_std"] > 0)
    if mask.any():
        # vectorised compute
        prices = pivot.loc[mask, price_cols].to_numpy(dtype=float)
        mu = pivot.loc[mask, "level_mean"].to_numpy(dtype=float).reshape(-1, 1)
        sd = pivot.loc[mask, "level_std"].to_numpy(dtype=float).reshape(-1, 1)
        z = (prices - mu) / sd
        pivot.loc[mask, shape_cols] = z

    # Add date-derived labels
    # Convert pivot["date"] (python date) to pandas Timestamp for weekday/month ops
    date_ts = pd.to_datetime(pivot["date"])
    pivot["daytype"] = date_ts.map(lambda x: daytype_of_date(x)).astype(str)

    season_cfg = SeasonConfig(winter_months=WINTER_MONTHS, summer_months=SUMMER_MONTHS)
    pivot["season"] = date_ts.dt.month.map(lambda m: season_of_month(int(m), season_cfg)).astype(str)

    pivot["class_id"] = pivot.apply(lambda r: conditioning_class(r["daytype"], r["season"]), axis=1)

    pivot["year"] = date_ts.dt.year.astype(int)
    pivot["split"] = pivot["year"].map(split_of_year).astype(str)

    return pivot


def main() -> int:
    in_path = Path(HARMONISED_HOURLY_PARQUET)
    out_dir = Path(DAILY_FEATURES_DIR)
    logs_root = Path(LOGS_ROOT)

    if not in_path.exists():
        raise FileNotFoundError(
            f"Missing hourly harmonised parquet:\n{in_path}\n"
            "Run Script 1 first (01_ingest_harmonise.py)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(logs_root)

    hourly = pd.read_parquet(in_path)

    # Basic schema check
    required = {"ts", "price_eur_mwh"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Hourly parquet missing columns: {missing}. Found: {list(hourly.columns)}")

    # Build daily features
    daily = build_daily_table(hourly)

    # Keep only years we care about (but keep out_of_scope rows for debugging if you want)
    # Here: write full file AND a scoped file.
    daily_scoped = daily[daily["split"].isin(["train", "validate", "test"])].copy()

    # Summary counts
    summary_by_split_class = (
        daily_scoped
        .groupby(["split", "class_id"], as_index=False)
        .agg(
            n_days=("date", "count"),
            n_complete=("is_complete_24h", "sum"),
            avg_missing_hours=("missing_hours", "mean"),
        )
        .sort_values(["split", "class_id"])
    )

    # Outputs
    out_daily_full = out_dir / "da_daily_features_full.parquet"
    out_daily_scoped = out_dir / "da_daily_features_scoped_2022_2025.parquet"
    out_summary = out_dir / "da_daily_summary_by_split_class.csv"

    daily.to_parquet(out_daily_full, index=False)
    daily_scoped.to_parquet(out_daily_scoped, index=False)
    summary_by_split_class.to_csv(out_summary, index=False, encoding="utf-8")

    # Manifest for reproducibility
    manifest: Dict = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "input_hourly_parquet": str(in_path),
        "output_dir": str(out_dir),
        "run_dir": str(run_dir),
        "split": {
            "train_years": TRAIN_YEARS,
            "validate_years": VALIDATE_YEARS,
            "test_years": TEST_YEARS,
        },
        "season_definition": {
            "winter_months": WINTER_MONTHS,
            "summer_months": SUMMER_MONTHS,
            "shoulder_months": [3, 4, 5, 9, 10],
        },
        "rows": {
            "hourly_rows_in": int(len(hourly)),
            "daily_rows_full": int(len(daily)),
            "daily_rows_scoped": int(len(daily_scoped)),
        },
        "quality": {
            "scoped_complete_days": int(daily_scoped["is_complete_24h"].sum()),
            "scoped_incomplete_days": int((~daily_scoped["is_complete_24h"]).sum()),
            "scoped_avg_missing_hours": float(daily_scoped["missing_hours"].mean()),
        },
        "outputs": {
            "daily_full_parquet": str(out_daily_full),
            "daily_scoped_parquet": str(out_daily_scoped),
            "summary_csv": str(out_summary),
        },
        "notes": [
            "Daily shape vectors are 24 hourly slots (00..23) in Europe/Amsterdam local time.",
            "Z-shapes are computed only when day is complete (24 hours) and daily std>0.",
            "Incomplete days are retained but flagged (DST/missing data). Exclude them from clustering unless you implement imputation.",
        ],
    }

    (run_dir / "02_manifest_daily_features.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Console output (PyCharm)
    print("=== DA Regimes: Script 2 Daily Features ===")
    print(f"Input hourly parquet : {in_path}")
    print(f"Output dir           : {out_dir}")
    print(f"Run dir              : {run_dir}")
    print(f"Daily rows (scoped)  : {len(daily_scoped)}")
    print(f"Complete days        : {int(daily_scoped['is_complete_24h'].sum())}")
    print(f"Incomplete days      : {int((~daily_scoped['is_complete_24h']).sum())}")
    print("Wrote:")
    print(f"  - {out_daily_scoped.name}")
    print(f"  - {out_summary.name}")
    print(f"Manifest: {run_dir / '02_manifest_daily_features.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
