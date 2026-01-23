from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------
# USER CONFIG
# ---------------------------

STRATEGY_A_DIR = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\03_regimes\strategy_A"
)

LOGS_ROOT = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\4_logs\da_regimes"
)

# Minimum empirical support in train for a regime (cluster) to be kept as a "normal" regime
MIN_CLUSTER_SIZE = 3

# Which splits define the stress-distance distribution for percentile computation
STRESS_SPLITS = ("train", "validate", "test")

# ---------------------------
# INTERNALS
# ---------------------------

def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def percentile_rank(values: np.ndarray, x: float) -> float:
    """
    Percentile rank in [0,1]. Uses <= convention.
    """
    if values.size == 0:
        return float("nan")
    return float(np.mean(values <= x))


def main() -> int:
    a_dir = Path(STRATEGY_A_DIR)
    logs_root = Path(LOGS_ROOT)
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(logs_root)

    # Inputs from Script 4
    p_assign = a_dir / "A_assignments_all_splits.parquet"
    p_probs = a_dir / "A_medoids_probabilities.parquet"
    p_medoids = a_dir / "A_medoids_shapes.parquet"
    p_stress = a_dir / "A_stress_library.parquet"
    p_k_by_class = a_dir / "A_k_by_class.json"

    for p in [p_assign, p_probs, p_medoids, p_stress, p_k_by_class]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required Strategy A artifact: {p}")

    assignments = pd.read_parquet(p_assign)
    probs = pd.read_parquet(p_probs)
    medoids = pd.read_parquet(p_medoids)
    stress = pd.read_parquet(p_stress)

    # ---------------------------
    # 1) Cluster support report (TRAIN)
    # ---------------------------
    train_assign = assignments[assignments["split"] == "train"].copy()

    support = (
        train_assign.groupby(["class_id", "cluster_id"], as_index=False)
        .agg(
            n_train=("date", "count"),
            mean_dist=("dist_to_medoid", "mean"),
            p90_dist=("dist_to_medoid", lambda x: float(np.quantile(x, 0.90))),
            max_dist=("dist_to_medoid", "max"),
        )
    )

    support["is_low_support"] = support["n_train"] < MIN_CLUSTER_SIZE
    support["is_singleton"] = support["n_train"] == 1

    support_summary = (
        support.groupby("class_id", as_index=False)
        .agg(
            k=("cluster_id", "nunique"),
            n_train_days=("n_train", "sum"),
            n_low_support=("is_low_support", "sum"),
            n_singletons=("is_singleton", "sum"),
            min_cluster_size=("n_train", "min"),
            median_cluster_size=("n_train", "median"),
            max_cluster_size=("n_train", "max"),
        )
        .sort_values("class_id")
    )

    out_support = a_dir / "A_cluster_support_train_by_class_cluster.csv"
    out_support_summary = a_dir / "A_cluster_support_train_summary_by_class.csv"
    support.sort_values(["class_id", "cluster_id"]).to_csv(out_support, index=False, encoding="utf-8")
    support_summary.to_csv(out_support_summary, index=False, encoding="utf-8")

    # ---------------------------
    # 2) Stress library percentiles (within class distance distribution)
    # ---------------------------
    stress_out = stress.copy()
    stress_out["pct_rank_within_class"] = np.nan

    for class_id in sorted(stress_out["class_id"].unique()):
        dist_pop = assignments[
            (assignments["class_id"] == class_id) & (assignments["split"].isin(STRESS_SPLITS))
        ]["dist_to_medoid"].to_numpy(dtype=float)

        mask = stress_out["class_id"] == class_id
        for idx in stress_out[mask].index:
            x = float(stress_out.loc[idx, "dist_to_medoid"])
            stress_out.loc[idx, "pct_rank_within_class"] = percentile_rank(dist_pop, x)

    out_stress_pct = a_dir / "A_stress_library_with_percentiles.csv"
    out_stress_pct_parquet = a_dir / "A_stress_library_with_percentiles.parquet"
    stress_out.to_csv(out_stress_pct, index=False, encoding="utf-8")
    stress_out.to_parquet(out_stress_pct_parquet, index=False)

    # ---------------------------
    # 3) Prune low-support regimes from probabilities (Strategy A QC)
    # ---------------------------
    # Identify regimes to KEEP as normal: n_train >= MIN_CLUSTER_SIZE
    keep = support[~support["is_low_support"]][["class_id", "cluster_id"]].copy()
    keep["keep_normal"] = True

    # Mark probs with keep flag
    probs_qc = probs.merge(
        keep,
        left_on=["class_id", "medoid_id"],
        right_on=["class_id", "cluster_id"],
        how="left",
    )
    probs_qc["keep_normal"] = probs_qc["keep_normal"].fillna(False)
    probs_qc.drop(columns=["cluster_id"], inplace=True)

    # Rare regimes = those not kept
    rare = probs_qc[~probs_qc["keep_normal"]].copy()
    rare = rare.rename(columns={"medoid_id": "cluster_id"})
    rare["reason"] = f"n_train < {MIN_CLUSTER_SIZE}"

    # Normal regimes: renormalize probabilities within each class after dropping rare ones
    normal = probs_qc[probs_qc["keep_normal"]].copy()

    # Sum of kept probability mass per class
    mass = normal.groupby("class_id", as_index=False)["p_medoid_given_class"].sum().rename(
        columns={"p_medoid_given_class": "kept_mass"}
    )
    normal = normal.merge(mass, on="class_id", how="left")

    # If kept_mass is 0 (pathological), keep original probabilities to avoid division by zero
    normal["p_medoid_given_class_pruned"] = np.where(
        normal["kept_mass"] > 0,
        normal["p_medoid_given_class"] / normal["kept_mass"],
        normal["p_medoid_given_class"],
    )

    # Output tables
    out_probs_qc = a_dir / "A_medoids_probabilities_qc.parquet"
    out_probs_qc_csv = a_dir / "A_medoids_probabilities_qc.csv"
    out_probs_pruned = a_dir / "A_medoids_probabilities_pruned.parquet"
    out_probs_pruned_csv = a_dir / "A_medoids_probabilities_pruned.csv"
    out_rare = a_dir / "A_rare_regimes.csv"

    probs_qc.to_parquet(out_probs_qc, index=False)
    probs_qc.to_csv(out_probs_qc_csv, index=False, encoding="utf-8")

    normal.to_parquet(out_probs_pruned, index=False)
    normal.to_csv(out_probs_pruned_csv, index=False, encoding="utf-8")

    rare.to_csv(out_rare, index=False, encoding="utf-8")

    # Also output rare medoid shapes for interpretability (dates + vectors)
    rare_shapes = rare.merge(
        medoids.rename(columns={"medoid_id": "cluster_id"})[["class_id", "cluster_id", "medoid_date"]],
        on=["class_id", "cluster_id"],
        how="left",
    )
    out_rare_shapes = a_dir / "A_rare_regimes_with_dates.csv"
    rare_shapes.to_csv(out_rare_shapes, index=False, encoding="utf-8")

    # ---------------------------
    # 4) Manifest
    # ---------------------------
    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "strategy_A_dir": str(a_dir),
        "run_dir": str(run_dir),
        "params": {
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "stress_splits_for_percentiles": list(STRESS_SPLITS),
        },
        "inputs": {
            "assignments": str(p_assign),
            "probabilities": str(p_probs),
            "medoids": str(p_medoids),
            "stress": str(p_stress),
            "k_by_class": str(p_k_by_class),
        },
        "outputs": {
            "cluster_support_by_class_cluster_csv": str(out_support),
            "cluster_support_summary_by_class_csv": str(out_support_summary),
            "stress_library_with_percentiles_csv": str(out_stress_pct),
            "probabilities_qc_csv": str(out_probs_qc_csv),
            "probabilities_pruned_csv": str(out_probs_pruned_csv),
            "rare_regimes_csv": str(out_rare),
            "rare_regimes_with_dates_csv": str(out_rare_shapes),
        },
        "notes": [
            "QC step flags low-support regimes (train cluster size < min_cluster_size).",
            "Pruned probabilities renormalize within each class after removing low-support regimes.",
            "Stress percentile ranks computed within each class distance distribution (selected splits).",
        ],
    }
    (run_dir / "04b_manifest_qc_prune_A.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Script 4b complete (QC + pruning Strategy A) ===")
    print(f"MIN_CLUSTER_SIZE = {MIN_CLUSTER_SIZE}")
    print(f"Wrote: {out_support_summary.name}")
    print(f"Wrote: {out_stress_pct.name}")
    print(f"Wrote: {out_probs_pruned_csv.name}")
    print(f"Wrote: {out_rare.name}")
    print(f"Manifest: {run_dir / '04b_manifest_qc_prune_A.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
