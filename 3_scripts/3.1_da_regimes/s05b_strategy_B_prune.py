from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# USER-TUNABLE PARAMETERS
# ---------------------------

# Minimum support thresholds to keep a regime as "normal"
MIN_RAW_N_TRAIN = 3          # raw count of train days assigned to the cluster
MIN_EFFECTIVE_N = 3.0        # Kish effective sample size under recency weights

# Which probability artifact to prune:
# - prefer weighted (if present) because that's what Strategy B actually uses
PREFER_WEIGHTED_PROBS = True


# ---------------------------
# PATH HELPERS
# ---------------------------

def find_repo_root(start: Path) -> Path:
    """
    Auto-detect repo root by searching upwards for '1_Configs' folder.
    """
    cur = start.resolve()
    for _ in range(12):
        if (cur / "1_Configs").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Could not find repo root (folder containing '1_Configs').")


def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


# ---------------------------
# STATS HELPERS
# ---------------------------

def kish_effective_n(weights: np.ndarray) -> float:
    """
    Kish effective sample size: (sum w)^2 / sum(w^2)
    """
    w = weights.astype(float)
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


def _pick_weight_column(df: pd.DataFrame) -> str:
    """
    Try to infer the recency weight column name.
    """
    candidates = [
        "w", "weight", "recency_weight", "w_recency", "w_final", "w_norm", "weight_norm"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not infer weight column. Available columns: {list(df.columns)}. "
        "Expected something like 'w' or 'recency_weight'."
    )


def _pick_prob_column(df: pd.DataFrame) -> str:
    """
    Try to infer the probability column name in medoid probabilities.
    """
    candidates = [
        "p_medoid_given_class",
        "p_medoid_given_class_weighted",
        "p",
        "prob",
        "probability",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not infer probability column. Available columns: {list(df.columns)}."
    )


def _pick_medoid_id_column(df: pd.DataFrame) -> str:
    """
    Try to infer medoid id column name.
    """
    candidates = ["medoid_id", "cluster_id", "prototype_id"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(
        f"Could not infer medoid id column. Available columns: {list(df.columns)}."
    )


# ---------------------------
# MAIN
# ---------------------------

def main() -> int:
    repo_root = find_repo_root(Path(__file__).parent)
    processed_root = repo_root / "1_Configs" / "1.2_Data_processed" / "DA-market" / "03_regimes"
    b_dir = processed_root / "strategy_B"
    if not b_dir.exists():
        raise FileNotFoundError(f"Strategy B folder not found: {b_dir}")

    logs_root = repo_root / "4_logs" / "da_regimes"
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(logs_root)

    # Inputs from Script 5 (Strategy B)
    p_assign_all = b_dir / "B_assignments_all_splits.parquet"
    p_medoids_shapes = b_dir / "B_medoids_shapes.parquet"
    p_weights = b_dir / "B_train_recency_weights.parquet"
    p_k_by_class = b_dir / "B_k_by_class.json"

    # probabilities: prefer weighted if present
    p_probs_weighted = b_dir / "B_medoids_probabilities_weighted.parquet"
    p_probs_base = b_dir / "B_medoids_probabilities.parquet"
    if PREFER_WEIGHTED_PROBS and p_probs_weighted.exists():
        p_probs = p_probs_weighted
        probs_kind = "weighted"
    elif p_probs_base.exists():
        p_probs = p_probs_base
        probs_kind = "base"
    else:
        raise FileNotFoundError(
            "No Strategy B probabilities found. Expected either "
            f"{p_probs_weighted.name} or {p_probs_base.name} in {b_dir}"
        )

    required = [p_assign_all, p_medoids_shapes, p_weights, p_k_by_class, p_probs]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f"Missing required Strategy B artifact: {p}")

    assignments = pd.read_parquet(p_assign_all)
    medoids = pd.read_parquet(p_medoids_shapes)
    weights = pd.read_parquet(p_weights)
    probs = pd.read_parquet(p_probs)

    # Basic schema checks
    for col in ["date", "split", "class_id", "cluster_id"]:
        if col not in assignments.columns:
            raise KeyError(f"assignments missing column '{col}'. Columns: {list(assignments.columns)}")

    if "dist_to_medoid" not in assignments.columns:
        # allow it, but we lose dist summaries
        pass

    if "date" not in weights.columns:
        raise KeyError(f"weights missing column 'date'. Columns: {list(weights.columns)}")

    wcol = _pick_weight_column(weights)
    # Ensure date types align for merge
    assignments["date"] = pd.to_datetime(assignments["date"]).dt.date
    weights["date"] = pd.to_datetime(weights["date"]).dt.date

    # Merge weights onto TRAIN assignments
    train_assign = assignments[assignments["split"] == "train"].copy()
    train_assign = train_assign.merge(weights[["date", wcol]], on="date", how="left")

    if train_assign[wcol].isna().any():
        n_missing = int(train_assign[wcol].isna().sum())
        raise ValueError(
            f"Found {n_missing} train assignment rows with missing recency weight after merge. "
            "This usually means the weight file doesn't cover all train dates."
        )

    # ---------------------------
    # 1) Cluster support report (TRAIN) — raw + weighted
    # ---------------------------
    group_cols = ["class_id", "cluster_id"]

    def _q(x: pd.Series, q: float) -> float:
        return float(np.quantile(x.to_numpy(dtype=float), q))

    agg_dict = {
        "n_train": ("date", "count"),
        "w_sum": (wcol, "sum"),
        "w_sq_sum": (wcol, lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))),
    }
    if "dist_to_medoid" in train_assign.columns:
        agg_dict.update(
            {
                "mean_dist": ("dist_to_medoid", "mean"),
                "p90_dist": ("dist_to_medoid", lambda x: _q(x, 0.90)),
                "max_dist": ("dist_to_medoid", "max"),
            }
        )

    support = (
        train_assign.groupby(group_cols, as_index=False)
        .agg(**{k: v for k, v in agg_dict.items()})
    )

    # Effective sample size under weights
    # n_eff = (sum w)^2 / sum(w^2)
    support["n_eff"] = np.where(
        support["w_sq_sum"] > 0.0,
        (support["w_sum"] * support["w_sum"]) / support["w_sq_sum"],
        0.0,
    )

    support["is_low_support_raw"] = support["n_train"] < MIN_RAW_N_TRAIN
    support["is_low_support_eff"] = support["n_eff"] < MIN_EFFECTIVE_N
    support["is_singleton_raw"] = support["n_train"] == 1

    support["keep_normal"] = ~(support["is_low_support_raw"] | support["is_low_support_eff"])

    # Summary per class
    support_summary = (
        support.groupby("class_id", as_index=False)
        .agg(
            k=("cluster_id", "nunique"),
            n_train_days=("n_train", "sum"),
            n_low_raw=("is_low_support_raw", "sum"),
            n_low_eff=("is_low_support_eff", "sum"),
            n_singletons=("is_singleton_raw", "sum"),
            min_cluster_size=("n_train", "min"),
            median_cluster_size=("n_train", "median"),
            max_cluster_size=("n_train", "max"),
            min_n_eff=("n_eff", "min"),
            median_n_eff=("n_eff", "median"),
            max_n_eff=("n_eff", "max"),
            kept_k=("keep_normal", "sum"),
        )
        .sort_values("class_id")
    )

    out_support = b_dir / "B_cluster_support_train_by_class_cluster.csv"
    out_support_summary = b_dir / "B_cluster_support_train_summary_by_class.csv"
    support.sort_values(["class_id", "cluster_id"]).to_csv(out_support, index=False, encoding="utf-8")
    support_summary.to_csv(out_support_summary, index=False, encoding="utf-8")

    # ---------------------------
    # 2) Prune low-support regimes from probabilities (Strategy B QC)
    # ---------------------------
    medoid_col = _pick_medoid_id_column(probs)
    prob_col = _pick_prob_column(probs)

    # Standardize a working copy with expected column names
    probs_qc = probs.copy()
    if medoid_col != "medoid_id":
        probs_qc = probs_qc.rename(columns={medoid_col: "medoid_id"})
    if prob_col != "p_medoid_given_class":
        probs_qc = probs_qc.rename(columns={prob_col: "p_medoid_given_class"})

    if "class_id" not in probs_qc.columns or "medoid_id" not in probs_qc.columns:
        raise KeyError(f"probabilities missing required keys. Columns: {list(probs_qc.columns)}")

    # Merge keep flags and support diagnostics into probs
    probs_qc = probs_qc.merge(
        support[
            ["class_id", "cluster_id", "n_train", "w_sum", "n_eff", "keep_normal",
             "is_low_support_raw", "is_low_support_eff"]
        ],
        left_on=["class_id", "medoid_id"],
        right_on=["class_id", "cluster_id"],
        how="left",
    )

    # If a medoid in probs isn't in support (shouldn't happen), mark as not-keep and flag
    probs_qc["keep_normal"] = probs_qc["keep_normal"].fillna(False)
    probs_qc["n_train"] = probs_qc["n_train"].fillna(0).astype(int)
    probs_qc["w_sum"] = probs_qc["w_sum"].fillna(0.0).astype(float)
    probs_qc["n_eff"] = probs_qc["n_eff"].fillna(0.0).astype(float)
    probs_qc["is_low_support_raw"] = probs_qc["is_low_support_raw"].fillna(True)
    probs_qc["is_low_support_eff"] = probs_qc["is_low_support_eff"].fillna(True)

    probs_qc = probs_qc.drop(columns=["cluster_id"])

    # Rare regimes table
    rare = probs_qc[~probs_qc["keep_normal"]].copy()
    rare = rare.rename(columns={"medoid_id": "cluster_id"})
    rare["reason"] = (
        f"low support: raw<{MIN_RAW_N_TRAIN} and/or eff<{MIN_EFFECTIVE_N}"
    )

    # Normal regimes: renormalize probabilities within each class after dropping rare ones
    normal = probs_qc[probs_qc["keep_normal"]].copy()

    mass = (
        normal.groupby("class_id", as_index=False)["p_medoid_given_class"]
        .sum()
        .rename(columns={"p_medoid_given_class": "kept_mass"})
    )
    normal = normal.merge(mass, on="class_id", how="left")

    normal["p_medoid_given_class_pruned"] = np.where(
        normal["kept_mass"] > 0.0,
        normal["p_medoid_given_class"] / normal["kept_mass"],
        normal["p_medoid_given_class"],
    )

    # ---------------------------
    # 3) Rare medoids with dates (for interpretability)
    # ---------------------------
    # Expect medoids file to have medoid_id and medoid_date, but stay robust
    medoids_work = medoids.copy()
    if "medoid_id" not in medoids_work.columns:
        # try typical alternatives
        for c in ["cluster_id", "prototype_id"]:
            if c in medoids_work.columns:
                medoids_work = medoids_work.rename(columns={c: "medoid_id"})
                break
    if "medoid_date" not in medoids_work.columns:
        # not fatal; we still output rare table
        pass

    rare_shapes = rare.copy()
    if "medoid_date" in medoids_work.columns and "medoid_id" in medoids_work.columns:
        rare_shapes = rare_shapes.merge(
            medoids_work[["class_id", "medoid_id", "medoid_date"]].rename(columns={"medoid_id": "cluster_id"}),
            on=["class_id", "cluster_id"],
            how="left",
        )

    # ---------------------------
    # 4) Write outputs (match A naming style)
    # ---------------------------
    out_probs_qc_pq = b_dir / "B_medoids_probabilities_qc.parquet"
    out_probs_qc_csv = b_dir / "B_medoids_probabilities_qc.csv"
    out_probs_pruned_pq = b_dir / "B_medoids_probabilities_pruned.parquet"
    out_probs_pruned_csv = b_dir / "B_medoids_probabilities_pruned.csv"
    out_rare = b_dir / "B_rare_regimes.csv"
    out_rare_dates = b_dir / "B_rare_regimes_with_dates.csv"

    probs_qc.to_parquet(out_probs_qc_pq, index=False)
    probs_qc.to_csv(out_probs_qc_csv, index=False, encoding="utf-8")

    normal.to_parquet(out_probs_pruned_pq, index=False)
    normal.to_csv(out_probs_pruned_csv, index=False, encoding="utf-8")

    rare.to_csv(out_rare, index=False, encoding="utf-8")
    rare_shapes.to_csv(out_rare_dates, index=False, encoding="utf-8")

    # ---------------------------
    # 5) Manifest
    # ---------------------------
    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "repo_root": str(repo_root),
        "strategy_B_dir": str(b_dir),
        "run_dir": str(run_dir),
        "params": {
            "min_raw_n_train": MIN_RAW_N_TRAIN,
            "min_effective_n": MIN_EFFECTIVE_N,
            "prefer_weighted_probs": PREFER_WEIGHTED_PROBS,
            "probabilities_used": str(p_probs),
            "probabilities_kind": probs_kind,
            "weight_column_used": wcol,
        },
        "inputs": {
            "assignments_all_splits": str(p_assign_all),
            "medoids_shapes": str(p_medoids_shapes),
            "train_recency_weights": str(p_weights),
            "k_by_class": str(p_k_by_class),
            "probabilities": str(p_probs),
        },
        "outputs": {
            "cluster_support_by_class_cluster_csv": str(out_support),
            "cluster_support_summary_by_class_csv": str(out_support_summary),
            "probabilities_qc_csv": str(out_probs_qc_csv),
            "probabilities_pruned_csv": str(out_probs_pruned_csv),
            "rare_regimes_csv": str(out_rare),
            "rare_regimes_with_dates_csv": str(out_rare_dates),
        },
        "notes": [
            "Strategy B pruning uses BOTH raw train support and weighted effective support (Kish n_eff).",
            "A regime is kept iff n_train >= MIN_RAW_N_TRAIN AND n_eff >= MIN_EFFECTIVE_N.",
            "Pruned probabilities renormalize within each class after removing low-support regimes.",
        ],
    }
    (run_dir / "05b_manifest_qc_prune_B.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # ---------------------------
    # 6) Console summary (quick QC)
    # ---------------------------
    print("=== Script 5b complete (QC + pruning Strategy B) ===")
    print(f"Probabilities used: {p_probs.name} ({probs_kind})")
    print(f"Thresholds: MIN_RAW_N_TRAIN={MIN_RAW_N_TRAIN}, MIN_EFFECTIVE_N={MIN_EFFECTIVE_N}")
    print(f"Wrote: {out_support_summary.name}")
    print(f"Wrote: {out_probs_pruned_csv.name}")
    print(f"Wrote: {out_rare.name}")
    print(f"Manifest: {run_dir / '05b_manifest_qc_prune_B.json'}")

    # QC: probability mass after pruning should be 1 per class
    mass_check = (
        normal.groupby("class_id")["p_medoid_given_class_pruned"]
        .sum()
        .sort_index()
    )
    worst = float(np.max(np.abs(mass_check.to_numpy() - 1.0))) if len(mass_check) else 0.0
    print(f"Max |sum(p_pruned)-1| across classes: {worst:.6g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
