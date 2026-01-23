from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# ---------------------------
# USER CONFIG
# ---------------------------

DAILY_FEATURES_PARQUET = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\01_daily_features"
    r"\da_daily_features_scoped_2022_2025.parquet"
)

# Optional: include dist_to_medoid context (not required)
ASSIGNMENTS_PARQUET = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\03_regimes\strategy_A"
    r"\A_assignments_all_splits.parquet"
)

OUT_DIR = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\03_regimes\strategy_A"
)

LOGS_ROOT = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\4_logs\da_regimes"
)

# Selection policy
TOP_PCT = 0.01
CAP_PER_CLASS = 3
SPLITS = ("train", "validate", "test")  # choose ("test",) if you only want extreme future stress

# Overall score weights
W_MEAN = 1.0 / 3.0
W_MAX = 1.0 / 3.0
W_STD = 1.0 / 3.0

# If you want "positive price stress only", set True (ignores negative price extremes)
POSITIVE_ONLY_FOR_MEAN_MAX = False


# ---------------------------
# INTERNALS
# ---------------------------

def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def safe_zscore(x: pd.Series) -> pd.Series:
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd


def select_top(df: pd.DataFrame, score_col: str, top_pct: float, cap: int) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return df.head(0)
    k = int(np.ceil(n * top_pct))
    k = max(1, k)
    return df.sort_values(score_col, ascending=False).head(k).head(cap)


def main() -> int:
    in_daily = Path(DAILY_FEATURES_PARQUET)
    in_assign = Path(ASSIGNMENTS_PARQUET)
    out_dir = Path(OUT_DIR)
    logs_root = Path(LOGS_ROOT)

    if not in_daily.exists():
        raise FileNotFoundError(f"Missing daily features: {in_daily}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(logs_root)

    daily = pd.read_parquet(in_daily)

    # We only want complete days for robust level metrics
    daily = daily[(daily["is_complete_24h"] == True) & (daily["split"].isin(SPLITS))].copy()

    required_cols = {"date", "split", "class_id", "level_mean", "level_max", "level_std"}
    missing = required_cols.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily features missing columns: {missing}")

    # Optional join: add dist_to_medoid for context
    if in_assign.exists():
        assign = pd.read_parquet(in_assign)
        assign = assign[assign["split"].isin(SPLITS)].copy()
        daily = daily.merge(
            assign[["date", "split", "class_id", "dist_to_medoid"]],
            on=["date", "split", "class_id"],
            how="left",
        )
    else:
        daily["dist_to_medoid"] = np.nan

    # Optionally focus on expensive days only for mean/max (keep std always)
    if POSITIVE_ONLY_FOR_MEAN_MAX:
        daily = daily.copy()
        daily.loc[daily["level_mean"] < 0, "level_mean"] = np.nan
        daily.loc[daily["level_max"] < 0, "level_max"] = np.nan

    # Compute within-class z-scores for level features
    daily["z_mean"] = daily.groupby("class_id")["level_mean"].transform(safe_zscore)
    daily["z_max"] = daily.groupby("class_id")["level_max"].transform(safe_zscore)
    daily["z_std"] = daily.groupby("class_id")["level_std"].transform(safe_zscore)

    # Combined score
    daily["stress_score"] = W_MEAN * daily["z_mean"] + W_MAX * daily["z_max"] + W_STD * daily["z_std"]

    # Percentile ranks within class for interpretability
    def pct_rank_within_class(x: pd.Series) -> pd.Series:
        arr = x.to_numpy()
        return x.rank(pct=True, method="average")  # 0..1 within group

    daily["pct_mean"] = daily.groupby("class_id")["level_mean"].transform(pct_rank_within_class)
    daily["pct_max"] = daily.groupby("class_id")["level_max"].transform(pct_rank_within_class)
    daily["pct_std"] = daily.groupby("class_id")["level_std"].transform(pct_rank_within_class)
    daily["pct_score"] = daily.groupby("class_id")["stress_score"].transform(pct_rank_within_class)

    # Build stress selections per class for each metric
    selections = []
    for class_id in sorted(daily["class_id"].unique()):
        sub = daily[daily["class_id"] == class_id].copy()

        # drop NaNs that could arise from POSITIVE_ONLY_FOR_MEAN_MAX
        sub_mean = sub.dropna(subset=["level_mean"])
        sub_max = sub.dropna(subset=["level_max"])
        sub_std = sub.dropna(subset=["level_std"])
        sub_score = sub.dropna(subset=["stress_score"])

        top_mean = select_top(sub_mean, "level_mean", TOP_PCT, CAP_PER_CLASS)
        top_max = select_top(sub_max, "level_max", TOP_PCT, CAP_PER_CLASS)
        top_std = select_top(sub_std, "level_std", TOP_PCT, CAP_PER_CLASS)
        top_score = select_top(sub_score, "stress_score", TOP_PCT, CAP_PER_CLASS)

        def tag(df_tag: pd.DataFrame, kind: str) -> pd.DataFrame:
            out = df_tag[[
                "date", "split", "class_id",
                "level_mean", "level_max", "level_std",
                "z_mean", "z_max", "z_std",
                "stress_score",
                "pct_mean", "pct_max", "pct_std", "pct_score",
                "dist_to_medoid",
            ]].copy()
            out["stress_kind"] = kind
            return out

        selections.append(tag(top_mean, "mean_price"))
        selections.append(tag(top_max, "max_price"))
        selections.append(tag(top_std, "volatility_std"))
        selections.append(tag(top_score, "overall_score"))

    stress_econ = pd.concat(selections, ignore_index=True).drop_duplicates(
        subset=["date", "split", "class_id", "stress_kind"]
    )

    # Also provide a compact per-class summary of thresholds (what is the cutoff for top 1%?)
    summary_rows = []
    for class_id in sorted(daily["class_id"].unique()):
        sub = daily[daily["class_id"] == class_id]
        n = len(sub)
        k = max(1, int(np.ceil(n * TOP_PCT)))
        # cutoffs are kth largest
        summary_rows.append({
            "class_id": class_id,
            "n_days": int(n),
            "top_pct": TOP_PCT,
            "k_raw": int(k),
            "cap": CAP_PER_CLASS,
            "cutoff_mean": float(sub["level_mean"].nlargest(k).min()) if sub["level_mean"].notna().any() else np.nan,
            "cutoff_max": float(sub["level_max"].nlargest(k).min()) if sub["level_max"].notna().any() else np.nan,
            "cutoff_std": float(sub["level_std"].nlargest(k).min()) if sub["level_std"].notna().any() else np.nan,
            "cutoff_score": float(sub["stress_score"].nlargest(k).min()) if sub["stress_score"].notna().any() else np.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values("class_id")

    # Write outputs
    out_csv = out_dir / "A_stress_library_economic.csv"
    out_pq = out_dir / "A_stress_library_economic.parquet"
    out_summary = out_dir / "A_stress_library_economic_cutoffs_by_class.csv"

    stress_econ.to_csv(out_csv, index=False, encoding="utf-8")
    stress_econ.to_parquet(out_pq, index=False)
    summary.to_csv(out_summary, index=False, encoding="utf-8")

    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "daily_features": str(in_daily),
            "assignments_optional": str(in_assign) if in_assign.exists() else None,
        },
        "params": {
            "top_pct": TOP_PCT,
            "cap_per_class": CAP_PER_CLASS,
            "splits": list(SPLITS),
            "weights": {"mean": W_MEAN, "max": W_MAX, "std": W_STD},
            "positive_only_for_mean_max": POSITIVE_ONLY_FOR_MEAN_MAX,
        },
        "outputs": {
            "economic_stress_csv": str(out_csv),
            "economic_stress_parquet": str(out_pq),
            "cutoffs_by_class_csv": str(out_summary),
        },
        "notes": [
            "Economic stress is defined by top percentile tails of level_mean, level_max, level_std and a combined z-score stress_score within each class.",
            "Percentiles are within-class for interpretability.",
            "dist_to_medoid is joined only for context; economic stress does not depend on clustering representation error.",
        ],
        "run_dir": str(run_dir),
    }
    (run_dir / "04c_manifest_economic_stress.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Script 4c complete (Economic stress library) ===")
    print(f"Wrote: {out_csv.name}")
    print(f"Wrote: {out_summary.name}")
    print(f"Manifest: {run_dir / '04c_manifest_economic_stress.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
