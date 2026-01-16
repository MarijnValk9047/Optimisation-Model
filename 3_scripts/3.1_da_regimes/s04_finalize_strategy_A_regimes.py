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

KNEE_SUGGESTIONS_JSON = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\02_clusters\elbow"
    r"\knee_suggestions.json"
)

REGIMES_DIR = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\03_regimes"
)

LOGS_ROOT = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\4_logs\da_regimes"
)

# PAM settings (must match Script 3 for consistency)
MAX_ITER = 200
RANDOM_SEED = 42

# Stress library settings
STRESS_TOP_PCT = 0.01
STRESS_CAP_PER_CLASS = 3

# Which splits to consider for stress selection
STRESS_SPLITS = ("train", "validate", "test")  # you can change to ("test",) if you want


# ---------------------------
# INTERNALS
# ---------------------------

SHAPE_COLS = [f"shape_z_h{h:02d}" for h in range(24)]


def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def extract_complete_split_class(df: pd.DataFrame, split: str, class_id: str) -> pd.DataFrame:
    sub = df[(df["split"] == split) & (df["class_id"] == class_id) & (df["is_complete_24h"] == True)].copy()
    sub = sub.dropna(subset=SHAPE_COLS)
    return sub


def zmat(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    dates = df["date"].to_numpy()
    Z = df[SHAPE_COLS].to_numpy(dtype=float)
    return dates, Z


def corr_distance_matrix_from_z(Z: np.ndarray) -> np.ndarray:
    G = (Z @ Z.T) / Z.shape[1]  # /24
    D = 1.0 - G
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def eval_representation(Z_eval: np.ndarray, Z_medoids: np.ndarray) -> np.ndarray:
    G = (Z_eval @ Z_medoids.T) / Z_eval.shape[1]
    D = 1.0 - G
    np.clip(D, 0.0, 2.0, out=D)
    return D.min(axis=1)


def assign_to_medoids_by_dot(Z: np.ndarray, Z_medoids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign each Z row to closest medoid using correlation distance (dot product).
    Returns (labels, dmin).
    """
    G = (Z @ Z_medoids.T) / Z.shape[1]
    D = 1.0 - G
    np.clip(D, 0.0, 2.0, out=D)
    labels = np.argmin(D, axis=1)
    dmin = D[np.arange(D.shape[0]), labels]
    return labels.astype(int), dmin.astype(float)


def init_medoids_greedy(D: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    N = D.shape[0]
    total_dist = D.sum(axis=1)
    first = int(np.argmin(total_dist))
    medoids = [first]
    dmin = D[:, first].copy()

    while len(medoids) < k:
        best_candidate = None
        best_improvement = -np.inf
        candidates = [i for i in range(N) if i not in medoids]
        rng.shuffle(candidates)
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
    rng = np.random.default_rng(seed)
    N = D.shape[0]

    medoid_idx = init_medoids_greedy(D, k, rng)
    # initial assignment
    dist_to_medoids = D[:, medoid_idx]
    labels = np.argmin(dist_to_medoids, axis=1)
    dmin = dist_to_medoids[np.arange(N), labels]
    best_cost = float(dmin.sum())

    improved = True
    it = 0

    while improved and it < max_iter:
        improved = False
        it += 1

        # recompute for current medoids
        medoid_set = set(map(int, medoid_idx))
        non_medoids = [i for i in range(N) if i not in medoid_set]

        dist_to_medoids = D[:, medoid_idx]
        current_dmin = dist_to_medoids.min(axis=1)
        current_argmin = dist_to_medoids.argmin(axis=1)

        best_swap = None
        best_swap_cost = best_cost

        for mi_pos, m in enumerate(medoid_idx):
            for h in non_medoids:
                d_to_h = D[:, h]
                mask_nearest_is_m = (current_argmin == mi_pos)

                if np.any(mask_nearest_is_m):
                    tmp = np.delete(dist_to_medoids, mi_pos, axis=1)
                    second_best = tmp.min(axis=1)
                else:
                    second_best = current_dmin

                new_dmin = current_dmin.copy()
                new_dmin[~mask_nearest_is_m] = np.minimum(current_dmin[~mask_nearest_is_m], d_to_h[~mask_nearest_is_m])
                new_dmin[mask_nearest_is_m] = np.minimum(second_best[mask_nearest_is_m], d_to_h[mask_nearest_is_m])

                new_cost = float(new_dmin.sum())
                if new_cost + 1e-9 < best_swap_cost:
                    best_swap_cost = new_cost
                    best_swap = (mi_pos, int(m), int(h))

        if best_swap is not None:
            mi_pos, m, h = best_swap
            medoid_idx = medoid_idx.copy()
            medoid_idx[mi_pos] = h
            medoid_idx = np.array(sorted(medoid_idx.tolist()), dtype=int)

            # update best cost and assignment
            dist_to_medoids = D[:, medoid_idx]
            labels = np.argmin(dist_to_medoids, axis=1)
            dmin = dist_to_medoids[np.arange(N), labels]
            best_cost = float(dmin.sum())

            improved = True

    return medoid_idx, labels.astype(int), dmin.astype(float), best_cost


def compute_tail_selection(df_assign: pd.DataFrame, top_pct: float, cap: int) -> pd.DataFrame:
    """
    df_assign must contain: date, split, class_id, dist_to_medoid
    Returns selected worst cases.
    """
    n = len(df_assign)
    if n == 0:
        return df_assign.head(0)
    k = int(np.ceil(n * top_pct))
    k = max(1, k)
    selected = df_assign.sort_values("dist_to_medoid", ascending=False).head(k)
    selected = selected.head(cap)
    return selected


def main() -> int:
    in_daily = Path(DAILY_FEATURES_PARQUET)
    in_knee = Path(KNEE_SUGGESTIONS_JSON)
    regimes_dir = Path(REGIMES_DIR)
    logs_root = Path(LOGS_ROOT)

    if not in_daily.exists():
        raise FileNotFoundError(f"Missing daily features parquet: {in_daily} (run Script 2).")
    if not in_knee.exists():
        raise FileNotFoundError(f"Missing knee_suggestions.json: {in_knee} (run Script 3).")

    regimes_dir.mkdir(parents=True, exist_ok=True)
    (regimes_dir / "strategy_A").mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(logs_root)

    df = pd.read_parquet(in_daily)
    knee: Dict[str, int] = json.loads(in_knee.read_text(encoding="utf-8"))

    classes = sorted(knee.keys())

    medoids_rows: List[Dict] = []
    probs_rows: List[Dict] = []
    assignments_rows: List[pd.DataFrame] = []
    eval_rows: List[Dict] = []
    stress_rows: List[pd.DataFrame] = []

    for class_id in classes:
        k = int(knee[class_id])

        train_df = extract_complete_split_class(df, "train", class_id)
        if len(train_df) <= k:
            raise ValueError(f"{class_id}: not enough train days ({len(train_df)}) for k={k}.")

        train_dates, Z_train = zmat(train_df)
        D_train = corr_distance_matrix_from_z(Z_train)

        med_idx, labels_train, dmin_train, cost = pam(D_train, k, MAX_ITER, RANDOM_SEED)
        medoid_dates = train_dates[med_idx]
        Z_medoids = Z_train[med_idx, :]

        # Store medoids (shapes)
        for j, (mi, d) in enumerate(zip(med_idx, medoid_dates)):
            row = {"class_id": class_id, "k": k, "medoid_id": j, "medoid_train_index": int(mi), "medoid_date": str(d)}
            for h, col in enumerate(SHAPE_COLS):
                row[col] = float(Z_medoids[j, h])
            medoids_rows.append(row)

        # Train probabilities (unweighted Strategy A): frequency of assigned cluster
        counts = pd.Series(labels_train).value_counts().sort_index()
        probs = (counts / counts.sum()).to_dict()
        for medoid_id in range(k):
            probs_rows.append(
                {
                    "class_id": class_id,
                    "k": k,
                    "medoid_id": int(medoid_id),
                    "p_medoid_given_class": float(probs.get(medoid_id, 0.0)),
                    "n_train_assigned": int(counts.get(medoid_id, 0)),
                }
            )

        # Assign train/val/test to nearest medoid (using dot distance, no need for full matrix)
        for split in ("train", "validate", "test"):
            sub = extract_complete_split_class(df, split, class_id)
            if len(sub) == 0:
                continue
            dates_sub, Z_sub = zmat(sub)
            labels, dmin = assign_to_medoids_by_dot(Z_sub, Z_medoids)

            assign_df = pd.DataFrame(
                {
                    "date": dates_sub,
                    "split": split,
                    "class_id": class_id,
                    "cluster_id": labels,
                    "dist_to_medoid": dmin,
                }
            )
            assignments_rows.append(assign_df)

            # Eval stats per split
            eval_rows.append(
                {
                    "class_id": class_id,
                    "k": k,
                    "split": split,
                    "n_days": int(len(assign_df)),
                    "mean_dist": float(np.mean(dmin)),
                    "p90_dist": float(np.quantile(dmin, 0.90)),
                    "p95_dist": float(np.quantile(dmin, 0.95)),
                    "max_dist": float(np.max(dmin)),
                }
            )

        # Stress selection within this class across chosen splits
        all_assign_class = pd.concat(
            [a for a in assignments_rows if (a["class_id"].iloc[0] == class_id and a["split"].iloc[0] in STRESS_SPLITS)],
            ignore_index=True,
        ) if assignments_rows else pd.DataFrame(columns=["date", "split", "class_id", "dist_to_medoid"])

        # If the above concat logic is too clever, fall back to filtering after global concat later.
        # We'll do robustly after loop as well; keep per-class selection simple:
        # We'll compute stress after we build global assignments.
        # (So do nothing here.)

        print(f"[{class_id}] k={k}  train_days={len(train_df)}  train_cost_mean={cost/len(train_df):.4f}")

    # Combine and write core tables
    medoids_df = pd.DataFrame(medoids_rows)
    probs_df = pd.DataFrame(probs_rows)
    eval_df = pd.DataFrame(eval_rows)
    assignments_df = pd.concat(assignments_rows, ignore_index=True) if assignments_rows else pd.DataFrame()

    out_medoids = regimes_dir / "strategy_A" / "A_medoids_shapes.parquet"
    out_probs = regimes_dir / "strategy_A" / "A_medoids_probabilities.parquet"
    out_eval = regimes_dir / "strategy_A" / "A_representation_error_by_split.parquet"
    out_assign = regimes_dir / "strategy_A" / "A_assignments_all_splits.parquet"

    medoids_df.to_parquet(out_medoids, index=False)
    probs_df.to_parquet(out_probs, index=False)
    eval_df.to_parquet(out_eval, index=False)
    assignments_df.to_parquet(out_assign, index=False)

    # Stress library: per class, pick top 1% worst distances within selected splits, cap 3
    stress_list = []
    for class_id, k in sorted({r["class_id"]: r["k"] for r in eval_rows}.items()):
        sub = assignments_df[(assignments_df["class_id"] == class_id) & (assignments_df["split"].isin(STRESS_SPLITS))].copy()
        sel = compute_tail_selection(sub, STRESS_TOP_PCT, STRESS_CAP_PER_CLASS)
        if len(sel) == 0:
            continue
        sel["k"] = int(k)
        stress_list.append(sel)

    stress_df = pd.concat(stress_list, ignore_index=True) if stress_list else pd.DataFrame()
    out_stress = regimes_dir / "strategy_A" / "A_stress_library.parquet"
    stress_df.to_parquet(out_stress, index=False)

    # Human-readable summary CSVs
    (regimes_dir / "strategy_A" / "A_k_by_class.json").write_text(json.dumps(knee, indent=2), encoding="utf-8")
    eval_csv = regimes_dir / "strategy_A" / "A_representation_error_by_split.csv"
    eval_df.sort_values(["class_id", "split"]).to_csv(eval_csv, index=False, encoding="utf-8")

    stress_csv = regimes_dir / "strategy_A" / "A_stress_library.csv"
    stress_df.to_csv(stress_csv, index=False, encoding="utf-8")

    # Manifest
    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "daily_features": str(in_daily),
            "knee_suggestions": str(in_knee),
        },
        "params": {
            "pam_max_iter": MAX_ITER,
            "seed": RANDOM_SEED,
            "stress_top_pct": STRESS_TOP_PCT,
            "stress_cap_per_class": STRESS_CAP_PER_CLASS,
            "stress_splits": list(STRESS_SPLITS),
        },
        "outputs": {
            "medoids_shapes_parquet": str(out_medoids),
            "medoids_probabilities_parquet": str(out_probs),
            "assignments_all_parquet": str(out_assign),
            "representation_error_parquet": str(out_eval),
            "stress_library_parquet": str(out_stress),
            "representation_error_csv": str(eval_csv),
            "stress_library_csv": str(stress_csv),
        },
        "notes": [
            "Strategy A uses unweighted train frequencies as probabilities p(medoid|class).",
            "Medoids are fit on train only; val/test are assigned to nearest train medoid.",
            "Stress library selects top 1% worst-represented days per class (by distance), capped at 3.",
        ],
        "run_dir": str(run_dir),
    }
    (run_dir / "04_manifest_strategy_A.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Script 4 complete (Strategy A frozen) ===")
    print(f"Wrote: {out_medoids}")
    print(f"Wrote: {out_probs}")
    print(f"Wrote: {out_assign}")
    print(f"Wrote: {out_eval}")
    print(f"Wrote: {out_stress}")
    print(f"Manifest: {run_dir / '04_manifest_strategy_A.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
