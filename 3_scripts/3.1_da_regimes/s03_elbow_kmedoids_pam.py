from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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

CLUSTERS_DIR = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\02_clusters"
)

LOGS_ROOT = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\4_logs\da_regimes"
)

LOCAL_TZ = "Europe/Amsterdam"

# Elbow sweep
K_MIN = 2
K_MAX = 20  # adjust if you want

# PAM settings
MAX_ITER = 200
RANDOM_SEED = 42

# Save per-k models (medoids + assignments). Turn off if you want fewer files.
SAVE_MODELS_PER_K = True


# ---------------------------
# HELPERS
# ---------------------------

SHAPE_COLS = [f"shape_z_h{h:02d}" for h in range(24)]


def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def knee_by_max_distance_to_chord(ks: np.ndarray, costs: np.ndarray) -> int:
    """
    Simple knee heuristic (common in elbow method):
    normalize ks and costs to [0,1], compute distance to line between endpoints,
    pick k with maximum distance.
    """
    if len(ks) < 3:
        return int(ks[-1])

    x = (ks - ks.min()) / (ks.max() - ks.min())
    y = (costs - costs.min()) / (costs.max() - costs.min() + 1e-12)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    chord = p2 - p1
    chord_norm = np.linalg.norm(chord) + 1e-12

    # distance from each point to chord line
    dists = []
    for xi, yi in zip(x, y):
        p = np.array([xi, yi])
        # area of parallelogram / base length
        v = p - p1
        cross_scalar = chord[0] * v[1] - chord[1] * v[0]  # 2D cross product
        dist = np.abs(cross_scalar) / chord_norm
        dists.append(dist)

    dists = np.array(dists)
    idx = int(np.argmax(dists))
    return int(ks[idx])


def extract_complete_split_class(df: pd.DataFrame, split: str, class_id: str) -> pd.DataFrame:
    sub = df[(df["split"] == split) & (df["class_id"] == class_id) & (df["is_complete_24h"] == True)].copy()
    # ensure shapes exist
    sub = sub.dropna(subset=SHAPE_COLS)
    return sub


def zmat(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (dates, Z) where:
      dates: array of python date objects
      Z: (N, 24) float array
    """
    dates = df["date"].to_numpy()
    Z = df[SHAPE_COLS].to_numpy(dtype=float)
    return dates, Z


def corr_distance_matrix_from_z(Z: np.ndarray) -> np.ndarray:
    """
    For z-normalised vectors (mean 0, std 1), correlation is proportional to dot product.
    corr ≈ (1/24) * Z @ Z.T
    distance = 1 - corr
    """
    G = (Z @ Z.T) / Z.shape[1]  # /24
    D = 1.0 - G
    # numerical safety
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def assign_to_medoids(D: np.ndarray, medoid_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Given full distance matrix D (N,N) and medoid indices, assigns each point to nearest medoid.
    Returns (labels, dmin, total_cost).
    """
    dist_to_medoids = D[:, medoid_idx]  # (N,k)
    labels = np.argmin(dist_to_medoids, axis=1)
    dmin = dist_to_medoids[np.arange(D.shape[0]), labels]
    total_cost = float(dmin.sum())
    return labels, dmin, total_cost


def init_medoids_greedy(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Greedy init:
      1) first medoid = point minimizing total distance
      2) iteratively add point that maximally reduces current assignment cost
    Deterministic except for tie-breaking.
    """
    N = D.shape[0]
    # first: best 1-medoids
    total_dist = D.sum(axis=1)
    first = int(np.argmin(total_dist))
    medoids = [first]

    # current best distance to a medoid
    dmin = D[:, first].copy()

    while len(medoids) < k:
        best_candidate = None
        best_improvement = -np.inf

        # try all candidates not already medoids
        candidates = [i for i in range(N) if i not in medoids]
        rng.shuffle(candidates)  # tie-breaking randomness
        for c in candidates:
            new_dmin = np.minimum(dmin, D[:, c])
            improvement = float(dmin.sum() - new_dmin.sum())
            if improvement > best_improvement:
                best_improvement = improvement
                best_candidate = c

        medoids.append(int(best_candidate))
        dmin = np.minimum(dmin, D[:, best_candidate])

    return np.array(medoids, dtype=int)


def pam(D: np.ndarray, k: int, max_iter: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Classic PAM (Partitioning Around Medoids):
    - greedy init
    - iterative swap medoid/non-medoid if improves objective
    Returns:
      medoid_idx, labels, dmin, total_cost
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]

    medoid_idx = init_medoids_greedy(D, k, rng)
    labels, dmin, best_cost = assign_to_medoids(D, medoid_idx)

    medoid_set = set(map(int, medoid_idx))
    non_medoids = [i for i in range(N) if i not in medoid_set]

    improved = True
    it = 0

    while improved and it < max_iter:
        improved = False
        it += 1

        # Try best swap in this iteration
        best_swap = None
        best_swap_cost = best_cost

        # For speed: precompute distances to current medoids
        dist_to_medoids = D[:, medoid_idx]  # (N,k)
        current_dmin = dist_to_medoids.min(axis=1)
        current_argmin = dist_to_medoids.argmin(axis=1)

        # Try swapping each medoid with each non-medoid
        for mi_pos, m in enumerate(medoid_idx):
            for h in non_medoids:
                # Compute new distances if we replace medoid m with h
                # For each point i:
                # - if its nearest medoid is not m, its dmin is min(current_dmin[i], D[i,h])
                # - if its nearest medoid is m, its new dmin is min(second_best_dist[i], D[i,h])
                # Compute second best dist efficiently
                # (small k -> cheap to compute by masking)
                d_to_h = D[:, h]
                mask_nearest_is_m = (current_argmin == mi_pos)

                # second best distances for points assigned to m
                # remove column mi_pos then take min
                if np.any(mask_nearest_is_m):
                    tmp = np.delete(dist_to_medoids, mi_pos, axis=1)  # (N, k-1)
                    second_best = tmp.min(axis=1)
                else:
                    second_best = current_dmin  # not used

                new_dmin = current_dmin.copy()
                # those not relying on m:
                new_dmin[~mask_nearest_is_m] = np.minimum(current_dmin[~mask_nearest_is_m], d_to_h[~mask_nearest_is_m])
                # those relying on m:
                new_dmin[mask_nearest_is_m] = np.minimum(second_best[mask_nearest_is_m], d_to_h[mask_nearest_is_m])

                new_cost = float(new_dmin.sum())
                if new_cost + 1e-9 < best_swap_cost:
                    best_swap_cost = new_cost
                    best_swap = (mi_pos, int(m), int(h))

        if best_swap is not None:
            mi_pos, m, h = best_swap
            # apply swap
            medoid_set.remove(m)
            medoid_set.add(h)
            medoid_idx = medoid_idx.copy()
            medoid_idx[mi_pos] = h
            medoid_idx = np.array(sorted(medoid_idx.tolist()), dtype=int)  # keep stable ordering

            # refresh non_medoids
            non_medoids = [i for i in range(N) if i not in set(map(int, medoid_idx))]

            labels, dmin, best_cost = assign_to_medoids(D, medoid_idx)
            improved = True

    return medoid_idx, labels, dmin, best_cost


def eval_representation(Z_eval: np.ndarray, Z_medoids: np.ndarray) -> np.ndarray:
    """
    Compute distance of each eval vector to nearest medoid using correlation distance:
      d = 1 - (1/24) * dot(z_eval, z_medoid)
    Returns dmin for each eval sample.
    """
    G = (Z_eval @ Z_medoids.T) / Z_eval.shape[1]
    D = 1.0 - G
    np.clip(D, 0.0, 2.0, out=D)
    dmin = D.min(axis=1)
    return dmin


def main() -> int:
    in_path = Path(DAILY_FEATURES_PARQUET)
    out_root = Path(CLUSTERS_DIR)
    logs_root = Path(LOGS_ROOT)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing daily features parquet: {in_path}\nRun Script 2 first.")

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "elbow").mkdir(parents=True, exist_ok=True)
    (out_root / "pam_models").mkdir(parents=True, exist_ok=True)

    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(logs_root)

    df = pd.read_parquet(in_path)

    # Classes present
    classes = sorted(df["class_id"].dropna().unique().tolist())

    results_rows: List[Dict] = []
    knee_suggestions: Dict[str, int] = {}

    for class_id in classes:
        train_df = extract_complete_split_class(df, "train", class_id)
        val_df = extract_complete_split_class(df, "validate", class_id)
        test_df = extract_complete_split_class(df, "test", class_id)

        if len(train_df) < K_MIN + 1:
            print(f"Skipping {class_id}: too few train days ({len(train_df)})")
            continue

        train_dates, Z_train = zmat(train_df)
        _, Z_val = zmat(val_df) if len(val_df) > 0 else (np.array([]), np.empty((0, 24)))
        _, Z_test = zmat(test_df) if len(test_df) > 0 else (np.array([]), np.empty((0, 24)))

        # Full distance matrix on train (needed for PAM objective)
        D_train = corr_distance_matrix_from_z(Z_train)

        ks = np.arange(K_MIN, min(K_MAX, len(train_df) - 1) + 1, dtype=int)
        costs_train = []

        class_out_dir = out_root / "pam_models" / class_id
        if SAVE_MODELS_PER_K:
            class_out_dir.mkdir(parents=True, exist_ok=True)

        for k in ks:
            med_idx, labels, dmin, cost = pam(D_train, int(k), MAX_ITER, RANDOM_SEED)
            costs_train.append(cost)

            # Prepare eval stats
            Z_medoids = Z_train[med_idx, :]
            dmin_val = eval_representation(Z_val, Z_medoids) if Z_val.shape[0] else np.array([])
            dmin_test = eval_representation(Z_test, Z_medoids) if Z_test.shape[0] else np.array([])

            def stats(arr: np.ndarray) -> Dict[str, float]:
                if arr.size == 0:
                    return {"mean": np.nan, "p90": np.nan, "p95": np.nan}
                return {
                    "mean": float(np.mean(arr)),
                    "p90": float(np.quantile(arr, 0.90)),
                    "p95": float(np.quantile(arr, 0.95)),
                }

            s_val = stats(dmin_val)
            s_test = stats(dmin_test)

            row = {
                "class_id": class_id,
                "k": int(k),
                "n_train": int(len(train_df)),
                "train_cost_sum": float(cost),
                "train_cost_mean": float(cost / len(train_df)),
                "val_mean_dist": s_val["mean"],
                "val_p90_dist": s_val["p90"],
                "val_p95_dist": s_val["p95"],
                "test_mean_dist": s_test["mean"],
                "test_p90_dist": s_test["p90"],
                "test_p95_dist": s_test["p95"],
            }
            results_rows.append(row)

            if SAVE_MODELS_PER_K:
                k_dir = class_out_dir / f"k_{int(k):02d}"
                k_dir.mkdir(parents=True, exist_ok=True)

                # Save medoids as dates (interpretable)
                medoid_dates = train_dates[med_idx]
                pd.DataFrame({"medoid_idx": med_idx, "medoid_date": medoid_dates}).to_csv(
                    k_dir / "medoids.csv", index=False, encoding="utf-8"
                )

                # Save train assignments
                assign_df = pd.DataFrame(
                    {
                        "date": train_dates,
                        "cluster_id": labels.astype(int),
                        "dist_to_medoid": dmin.astype(float),
                    }
                )
                assign_df.to_parquet(k_dir / "assignments_train.parquet", index=False)

        # Write per-class elbow table
        elbow_df = pd.DataFrame([r for r in results_rows if r["class_id"] == class_id]).sort_values("k")
        elbow_path = out_root / "elbow" / f"elbow_{class_id}.csv"
        elbow_df.to_csv(elbow_path, index=False, encoding="utf-8")

        # Knee suggestion from training cost curve
        costs = elbow_df["train_cost_sum"].to_numpy(dtype=float)
        k_suggest = knee_by_max_distance_to_chord(elbow_df["k"].to_numpy(dtype=int), costs)
        knee_suggestions[class_id] = int(k_suggest)

        print(f"[{class_id}] train days={len(train_df)}  k_range={ks[0]}..{ks[-1]}  knee_suggest={k_suggest}")

    # Combined elbow table
    all_elbow = pd.DataFrame(results_rows).sort_values(["class_id", "k"])
    combined_path = out_root / "elbow" / "elbow_all_classes.csv"
    all_elbow.to_csv(combined_path, index=False, encoding="utf-8")

    # Knee suggestions
    knee_path = out_root / "elbow" / "knee_suggestions.json"
    knee_path.write_text(json.dumps(knee_suggestions, indent=2), encoding="utf-8")

    # Manifest
    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "input_daily_features": str(in_path),
        "output_root": str(out_root),
        "run_dir": str(run_dir),
        "k_range": {"k_min": K_MIN, "k_max": K_MAX},
        "pam": {"max_iter": MAX_ITER, "seed": RANDOM_SEED},
        "saved_models_per_k": SAVE_MODELS_PER_K,
        "classes": list(knee_suggestions.keys()),
        "knee_suggestions": knee_suggestions,
        "outputs": {
            "elbow_all_classes_csv": str(combined_path),
            "knee_suggestions_json": str(knee_path),
        },
        "notes": [
            "Clustering uses train split only; val/test are evaluated against train medoids.",
            "Distance is correlation distance on 24-dim z-shape vectors.",
            "Incomplete days were excluded (is_complete_24h=True and non-null shapes).",
            "Knee suggestion uses max distance-to-chord heuristic on train_cost_sum.",
        ],
    }
    (run_dir / "03_manifest_elbow_kmedoids.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Script 3 complete ===")
    print(f"Wrote: {combined_path}")
    print(f"Wrote: {knee_path}")
    print(f"Manifest: {run_dir / '03_manifest_elbow_kmedoids.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
