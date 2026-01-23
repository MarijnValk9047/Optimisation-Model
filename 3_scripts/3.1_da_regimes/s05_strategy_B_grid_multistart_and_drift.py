from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ============================
# USER CONFIG
# ============================

DAILY_FEATURES_PARQUET = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\01_daily_features"
    r"\da_daily_features_scoped_2022_2025.parquet"
)

K_BY_CLASS_JSON = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\03_regimes\strategy_A"
    r"\A_k_by_class.json"
)

OUT_DIR = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\1_Configs\1.2_Data_processed\DA-market\03_regimes\strategy_B"
)

LOGS_ROOT = (
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model"
    r"\4_logs\da_regimes"
)

# Strategy B recency weighting
BETA = 0.5
HALF_LIFE_GRID_MONTHS = [6, 12, 18, 24, 36, 48]  # grid search on validation
REFERENCE_DATE_MODE = "train_end"  # "train_end" recommended for your design

# Multi-start settings
N_STARTS = 7
MAX_ITER = 200
BASE_SEED = 42

# Validation selection criterion (validate only)
# Primary: minimize val_p95_dist; tie-break: val_mean_dist; then val_p90_dist
# You can switch to a scalar score below if you prefer.
USE_SCALAR_VALID_SCORE = False
VALID_SCORE_WEIGHTS = {"mean": 0.3, "p95": 0.7}  # used only if USE_SCALAR_VALID_SCORE=True

# Drift testing settings
PERMUTATIONS = 500  # increase to 1000+ for thesis runs if runtime allows
RBF_BANDWIDTH = "median"  # "median" heuristic
DRIFT_PAIRS = [("train", "validate"), ("train", "test"), ("validate", "test")]

# Day completeness
REQUIRE_COMPLETE_24H = True

# ============================
# INTERNALS
# ============================

SHAPE_COLS = [f"shape_z_h{h:02d}" for h in range(24)]
LEVEL_COLS = ["level_mean", "level_max", "level_std"]


def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def months_between(d1: pd.Timestamp, d2: pd.Timestamp) -> float:
    # approximate month length 30.4375 days (365.25/12)
    return float((d2 - d1).days) / 30.4375


def recency_weights(dates: np.ndarray, reference_date: pd.Timestamp, half_life_months: float) -> np.ndarray:
    """
    w = 2^(-age/half_life) with age in months, where age = (reference_date - date)
    """
    dts = pd.to_datetime(dates)
    ages = np.array([months_between(pd.Timestamp(dt), reference_date) for dt in dts], dtype=float)
    ages = np.maximum(0.0, ages)
    w = np.power(2.0, -ages / float(half_life_months))
    return w


def corr_distance_matrix_from_z(Z: np.ndarray) -> np.ndarray:
    # correlation distance given z-normalised daily shapes
    G = (Z @ Z.T) / Z.shape[1]
    D = 1.0 - G
    np.clip(D, 0.0, 2.0, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def assign_to_medoids_by_dot(Z: np.ndarray, Z_medoids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    G = (Z @ Z_medoids.T) / Z.shape[1]
    D = 1.0 - G
    np.clip(D, 0.0, 2.0, out=D)
    labels = np.argmin(D, axis=1).astype(int)
    dmin = D[np.arange(D.shape[0]), labels].astype(float)
    return labels, dmin


def weighted_assign_cost(D: np.ndarray, medoid_idx: np.ndarray, w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    dist_to_medoids = D[:, medoid_idx]
    labels = np.argmin(dist_to_medoids, axis=1)
    dmin = dist_to_medoids[np.arange(D.shape[0]), labels]
    cost = float(np.sum(w * dmin))
    return labels.astype(int), dmin.astype(float), cost


def init_medoids_greedy_weighted(D: np.ndarray, k: int, w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Weighted greedy init:
    - first medoid minimizes sum_i w_i * D[i,m]
    - then iteratively add candidate maximizing reduction in weighted assignment cost
    """
    N = D.shape[0]
    weighted_total = (D * w[:, None]).sum(axis=0)  # cost if single medoid = m
    first = int(np.argmin(weighted_total))
    medoids = [first]

    dmin = D[:, first].copy()
    cost = float(np.sum(w * dmin))

    while len(medoids) < k:
        best_c = None
        best_improvement = -np.inf

        candidates = [i for i in range(N) if i not in medoids]
        rng.shuffle(candidates)
        for c in candidates:
            new_dmin = np.minimum(dmin, D[:, c])
            new_cost = float(np.sum(w * new_dmin))
            improvement = cost - new_cost
            if improvement > best_improvement:
                best_improvement = improvement
                best_c = c

        medoids.append(int(best_c))
        dmin = np.minimum(dmin, D[:, best_c])
        cost = float(np.sum(w * dmin))

    return np.array(medoids, dtype=int)


def pam_weighted(D: np.ndarray, k: int, w: np.ndarray, max_iter: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Weighted PAM:
    objective = sum_i w_i * dist(i, nearest_medoid)
    """
    rng = np.random.default_rng(seed)
    N = D.shape[0]

    medoid_idx = init_medoids_greedy_weighted(D, k, w, rng)
    labels, dmin, best_cost = weighted_assign_cost(D, medoid_idx, w)

    improved = True
    it = 0

    while improved and it < max_iter:
        improved = False
        it += 1

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
                    second_best = current_dmin  # not used

                new_dmin = current_dmin.copy()
                new_dmin[~mask_nearest_is_m] = np.minimum(current_dmin[~mask_nearest_is_m], d_to_h[~mask_nearest_is_m])
                new_dmin[mask_nearest_is_m] = np.minimum(second_best[mask_nearest_is_m], d_to_h[mask_nearest_is_m])

                new_cost = float(np.sum(w * new_dmin))
                if new_cost + 1e-9 < best_swap_cost:
                    best_swap_cost = new_cost
                    best_swap = (mi_pos, int(m), int(h))

        if best_swap is not None:
            mi_pos, _, h = best_swap
            medoid_idx = medoid_idx.copy()
            medoid_idx[mi_pos] = h
            medoid_idx = np.array(sorted(medoid_idx.tolist()), dtype=int)

            labels, dmin, best_cost = weighted_assign_cost(D, medoid_idx, w)
            improved = True

    return medoid_idx, labels.astype(int), dmin.astype(float), float(best_cost)


def jaccard(a: List[str], b: List[str]) -> float:
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def extract(df: pd.DataFrame, split: str, class_id: str) -> pd.DataFrame:
    sub = df[(df["split"] == split) & (df["class_id"] == class_id)].copy()
    if REQUIRE_COMPLETE_24H:
        sub = sub[sub["is_complete_24h"] == True].copy()
    sub = sub.dropna(subset=SHAPE_COLS + LEVEL_COLS)
    return sub


def energy_stat_from_dist(D: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
    # Energy distance estimator: 2 E|X-Y| - E|X-X'| - E|Y-Y'| with finite-sample means excluding diagonal
    x = idx_x
    y = idx_y

    d_xy = D[np.ix_(x, y)]
    e_xy = float(d_xy.mean()) if d_xy.size else 0.0

    d_xx = D[np.ix_(x, x)]
    d_yy = D[np.ix_(y, y)]

    def mean_offdiag(A: np.ndarray) -> float:
        n = A.shape[0]
        if n <= 1:
            return 0.0
        return float((A.sum() - np.trace(A)) / (n * (n - 1)))

    e_xx = mean_offdiag(d_xx)
    e_yy = mean_offdiag(d_yy)
    return float(2.0 * e_xy - e_xx - e_yy)


def rbf_kernel_matrix(X: np.ndarray, gamma: float) -> np.ndarray:
    # K_ij = exp(-gamma * ||xi-xj||^2)
    # compute squared distances efficiently
    X2 = np.sum(X * X, axis=1, keepdims=True)
    dist2 = X2 + X2.T - 2.0 * (X @ X.T)
    np.maximum(dist2, 0.0, out=dist2)
    return np.exp(-gamma * dist2)


def mmd2_stat_from_kernel(K: np.ndarray, idx_x: np.ndarray, idx_y: np.ndarray) -> float:
    # unbiased MMD^2 using off-diagonal means
    x = idx_x
    y = idx_y
    K_xx = K[np.ix_(x, x)]
    K_yy = K[np.ix_(y, y)]
    K_xy = K[np.ix_(x, y)]

    def mean_offdiag(A: np.ndarray) -> float:
        n = A.shape[0]
        if n <= 1:
            return 0.0
        return float((A.sum() - np.trace(A)) / (n * (n - 1)))

    m_xx = mean_offdiag(K_xx)
    m_yy = mean_offdiag(K_yy)
    m_xy = float(K_xy.mean()) if K_xy.size else 0.0
    return float(m_xx + m_yy - 2.0 * m_xy)


def permutation_pvalue(stat_obs: float, stats_perm: np.ndarray) -> float:
    # one-sided: larger stat => more drift
    return float((1.0 + np.sum(stats_perm >= stat_obs)) / (1.0 + len(stats_perm)))


def drift_tests_for_class(X: np.ndarray, y_splits: np.ndarray, splits: Tuple[str, str], rng: np.random.Generator) -> Dict[str, float]:
    """
    X: samples (N,d)
    y_splits: array of split labels length N
    splits: (a,b)
    Returns energy_stat, energy_p, mmd_stat, mmd_p
    """
    a, b = splits
    idx_a = np.where(y_splits == a)[0]
    idx_b = np.where(y_splits == b)[0]

    if len(idx_a) < 2 or len(idx_b) < 2:
        return {"n_a": float(len(idx_a)), "n_b": float(len(idx_b)),
                "energy": np.nan, "p_energy": np.nan, "mmd2": np.nan, "p_mmd2": np.nan}

    # pooled distances for energy
    # Euclidean distances
    X2 = np.sum(X * X, axis=1, keepdims=True)
    dist2 = X2 + X2.T - 2.0 * (X @ X.T)
    np.maximum(dist2, 0.0, out=dist2)
    D = np.sqrt(dist2)

    energy_obs = energy_stat_from_dist(D, idx_a, idx_b)

    # MMD kernel bandwidth
    if RBF_BANDWIDTH == "median":
        # median of off-diagonal distances
        tri = D[np.triu_indices(D.shape[0], k=1)]
        med = float(np.median(tri)) if tri.size else 1.0
        sigma = med if med > 1e-12 else 1.0
    else:
        sigma = float(RBF_BANDWIDTH)
    gamma = 1.0 / (2.0 * sigma * sigma)

    K = rbf_kernel_matrix(X, gamma)
    mmd_obs = mmd2_stat_from_kernel(K, idx_a, idx_b)

    # permutations (keep sizes)
    n_a = len(idx_a)
    all_idx = np.arange(len(y_splits))

    e_perm = np.empty(PERMUTATIONS, dtype=float)
    m_perm = np.empty(PERMUTATIONS, dtype=float)

    for p in range(PERMUTATIONS):
        perm = rng.permutation(all_idx)
        perm_a = perm[:n_a]
        perm_b = perm[n_a:]
        e_perm[p] = energy_stat_from_dist(D, perm_a, perm_b)
        m_perm[p] = mmd2_stat_from_kernel(K, perm_a, perm_b)

    return {
        "n_a": float(n_a),
        "n_b": float(len(idx_b)),
        "energy": float(energy_obs),
        "p_energy": permutation_pvalue(energy_obs, e_perm),
        "mmd2": float(mmd_obs),
        "p_mmd2": permutation_pvalue(mmd_obs, m_perm),
        "sigma_rbf": float(sigma),
    }


def split_eval_stats(dmin: np.ndarray) -> Dict[str, float]:
    if dmin.size == 0:
        return {"mean": np.nan, "p90": np.nan, "p95": np.nan, "max": np.nan}
    return {
        "mean": float(np.mean(dmin)),
        "p90": float(np.quantile(dmin, 0.90)),
        "p95": float(np.quantile(dmin, 0.95)),
        "max": float(np.max(dmin)),
    }


def choose_best_half_life(grid_rows: pd.DataFrame) -> int:
    """
    grid_rows: rows for ONE class_id with columns:
      half_life_months, val_mean_dist, val_p90_dist, val_p95_dist
    """
    g = grid_rows.copy()

    if USE_SCALAR_VALID_SCORE:
        g["valid_score"] = (
            VALID_SCORE_WEIGHTS["mean"] * g["val_mean_dist"] +
            VALID_SCORE_WEIGHTS["p95"] * g["val_p95_dist"]
        )
        g = g.sort_values(["valid_score", "val_p90_dist", "half_life_months"], ascending=[True, True, True])
        return int(g.iloc[0]["half_life_months"])

    # lexicographic: p95, then mean, then p90, then smallest half-life
    g = g.sort_values(["val_p95_dist", "val_mean_dist", "val_p90_dist", "half_life_months"],
                      ascending=[True, True, True, True])
    return int(g.iloc[0]["half_life_months"])


def main() -> int:
    in_daily = Path(DAILY_FEATURES_PARQUET)
    in_k = Path(K_BY_CLASS_JSON)
    out_dir = Path(OUT_DIR)
    logs_root = Path(LOGS_ROOT)

    if not in_daily.exists():
        raise FileNotFoundError(f"Missing daily features: {in_daily}")
    if not in_k.exists():
        raise FileNotFoundError(f"Missing k_by_class json: {in_k} (from Strategy A outputs)")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "grid").mkdir(parents=True, exist_ok=True)
    (out_dir / "drift").mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)
    run_dir = make_run_dir(logs_root)

    df = pd.read_parquet(in_daily)
    k_by_class: Dict[str, int] = json.loads(in_k.read_text(encoding="utf-8"))
    classes = sorted(k_by_class.keys())

    # reference date
    if REFERENCE_DATE_MODE == "train_end":
        ref_date = pd.Timestamp("2023-12-31")
    else:
        ref_date = pd.to_datetime(df["date"]).max()

    # ============================
    # PART A: Drift tests (shape + level) with permutation p-values
    # ============================
    rng_drift = np.random.default_rng(BASE_SEED + 1000)

    drift_rows: List[Dict] = []

    for class_id in classes:
        # pooled (train/val/test) complete days
        pooled = df[df["class_id"] == class_id].copy()
        if REQUIRE_COMPLETE_24H:
            pooled = pooled[pooled["is_complete_24h"] == True].copy()
        pooled = pooled.dropna(subset=SHAPE_COLS + LEVEL_COLS + ["split"])

        if len(pooled) == 0:
            continue

        y_splits = pooled["split"].to_numpy()
        # shape drift in 24D
        X_shape = pooled[SHAPE_COLS].to_numpy(dtype=float)
        # level drift in 3D (mean,max,std)
        X_level = pooled[LEVEL_COLS].to_numpy(dtype=float)

        for a, b in DRIFT_PAIRS:
            # shape drift
            res_s = drift_tests_for_class(X_shape, y_splits, (a, b), rng_drift)
            drift_rows.append({
                "class_id": class_id,
                "feature_space": "shape_24D",
                "split_a": a,
                "split_b": b,
                **res_s,
            })
            # level drift
            res_l = drift_tests_for_class(X_level, y_splits, (a, b), rng_drift)
            drift_rows.append({
                "class_id": class_id,
                "feature_space": "level_3D",
                "split_a": a,
                "split_b": b,
                **res_l,
            })

    drift_df = pd.DataFrame(drift_rows)
    drift_out = out_dir / "drift" / "B_drift_tests_energy_mmd_permutation.csv"
    drift_df.to_csv(drift_out, index=False, encoding="utf-8")

    # ============================
    # PART B: Half-life grid + multi-start weighted PAM (validate-only selection)
    # ============================
    grid_rows: List[Dict] = []
    stability_rows: List[Dict] = []

    # store best model for each class and half-life (medoids indices + dates)
    best_models: Dict[Tuple[str, int], Dict] = {}

    for class_id in classes:
        k = int(k_by_class[class_id])

        train_df = extract(df, "train", class_id)
        val_df = extract(df, "validate", class_id)
        test_df = extract(df, "test", class_id)

        if len(train_df) <= k:
            raise ValueError(f"{class_id}: insufficient train days ({len(train_df)}) for k={k}")

        train_dates = pd.to_datetime(train_df["date"]).to_numpy()
        Z_train = train_df[SHAPE_COLS].to_numpy(dtype=float)
        D_train = corr_distance_matrix_from_z(Z_train)

        # prepare val/test matrices
        val_dates = pd.to_datetime(val_df["date"]).to_numpy() if len(val_df) else np.array([])
        Z_val = val_df[SHAPE_COLS].to_numpy(dtype=float) if len(val_df) else np.empty((0, 24))
        test_dates = pd.to_datetime(test_df["date"]).to_numpy() if len(test_df) else np.array([])
        Z_test = test_df[SHAPE_COLS].to_numpy(dtype=float) if len(test_df) else np.empty((0, 24))

        for hl in HALF_LIFE_GRID_MONTHS:
            # weights on training days
            w = recency_weights(train_dates, ref_date, float(hl))
            w = np.power(w, BETA)  # beta tempering
            w = w / (np.sum(w) + 1e-12)

            # multi-start
            start_medoids: List[List[str]] = []
            start_costs: List[float] = []

            best = None
            best_cost = np.inf
            best_seed = None
            best_med_idx = None

            for s in range(N_STARTS):
                seed = BASE_SEED + 10_000 * hl + 100 * s
                med_idx, labels_tr, dmin_tr, cost = pam_weighted(D_train, k, w, MAX_ITER, seed)
                start_costs.append(cost)
                med_dates = [str(pd.Timestamp(d).date()) for d in train_dates[med_idx]]
                start_medoids.append(med_dates)

                if cost < best_cost:
                    best_cost = cost
                    best = (labels_tr, dmin_tr)
                    best_seed = seed
                    best_med_idx = med_idx

            # stability: how similar are medoid sets to best across starts?
            best_set = start_medoids[int(np.argmin(start_costs))]
            jaccs = [jaccard(best_set, m) for m in start_medoids]
            stability_rows.append({
                "class_id": class_id,
                "half_life_months": int(hl),
                "k": k,
                "n_starts": N_STARTS,
                "best_cost_weighted": float(best_cost),
                "mean_cost_weighted": float(np.mean(start_costs)),
                "std_cost_weighted": float(np.std(start_costs)),
                "jaccard_to_best_mean": float(np.mean(jaccs)),
                "jaccard_to_best_min": float(np.min(jaccs)),
                "best_seed": int(best_seed),
            })

            # evaluate best medoids on val/test
            Z_medoids = Z_train[best_med_idx, :]
            val_labels, val_dmin = assign_to_medoids_by_dot(Z_val, Z_medoids) if Z_val.shape[0] else (np.array([]), np.array([]))
            test_labels, test_dmin = assign_to_medoids_by_dot(Z_test, Z_medoids) if Z_test.shape[0] else (np.array([]), np.array([]))

            s_val = split_eval_stats(val_dmin.astype(float))
            s_test = split_eval_stats(test_dmin.astype(float))

            grid_rows.append({
                "class_id": class_id,
                "k": k,
                "half_life_months": int(hl),
                "beta": float(BETA),
                "ref_date": str(ref_date.date()),
                "train_cost_weighted_best": float(best_cost),
                "val_mean_dist": s_val["mean"],
                "val_p90_dist": s_val["p90"],
                "val_p95_dist": s_val["p95"],
                "val_max_dist": s_val["max"],
                "test_mean_dist": s_test["mean"],
                "test_p90_dist": s_test["p90"],
                "test_p95_dist": s_test["p95"],
                "test_max_dist": s_test["max"],
            })

            # store best model for later freezing
            best_models[(class_id, int(hl))] = {
                "best_medoid_idx": best_med_idx,
                "train_dates": train_dates,
                "Z_train": Z_train,
                "weights": w,
            }

    grid_df = pd.DataFrame(grid_rows).sort_values(["class_id", "half_life_months"])
    stability_df = pd.DataFrame(stability_rows).sort_values(["class_id", "half_life_months"])

    grid_out = out_dir / "grid" / "B_half_life_grid_validate.csv"
    stability_out = out_dir / "grid" / "B_multistart_stability.csv"
    grid_df.to_csv(grid_out, index=False, encoding="utf-8")
    stability_df.to_csv(stability_out, index=False, encoding="utf-8")

    # pick best half-life per class using validate only
    best_hl_by_class: Dict[str, int] = {}
    for class_id in classes:
        g = grid_df[grid_df["class_id"] == class_id].copy()
        best_hl_by_class[class_id] = choose_best_half_life(g)

    best_hl_path = out_dir / "grid" / "B_best_half_life_by_class.json"
    best_hl_path.write_text(json.dumps(best_hl_by_class, indent=2), encoding="utf-8")

    # ============================
    # PART C: Freeze final Strategy B using selected half-lives
    # ============================
    medoids_rows: List[Dict] = []
    probs_rows: List[Dict] = []
    assignments_rows: List[pd.DataFrame] = []
    eval_rows: List[Dict] = []

    for class_id in classes:
        hl = int(best_hl_by_class[class_id])
        k = int(k_by_class[class_id])

        model = best_models[(class_id, hl)]
        best_med_idx = model["best_medoid_idx"]
        train_dates = model["train_dates"]
        Z_train = model["Z_train"]
        w = model["weights"]

        Z_medoids = Z_train[best_med_idx, :]
        medoid_dates = train_dates[best_med_idx]

        # store medoids (shape vectors)
        for j, d in enumerate(medoid_dates):
            row = {
                "class_id": class_id,
                "k": k,
                "half_life_months": hl,
                "beta": float(BETA),
                "ref_date": str(ref_date.date()),
                "medoid_id": int(j),
                "medoid_date": str(pd.Timestamp(d).date()),
            }
            for h, col in enumerate(SHAPE_COLS):
                row[col] = float(Z_medoids[j, h])
            medoids_rows.append(row)

        # compute weighted probabilities based on weighted mass of assigned days (TRAIN)
        # We need labels on train for the chosen medoids:
        D_train = corr_distance_matrix_from_z(Z_train)
        labels_tr, dmin_tr, cost_tr = weighted_assign_cost(D_train, best_med_idx, w)
        # weighted mass per cluster
        mass = pd.Series(w).groupby(labels_tr).sum()
        mass = mass.reindex(range(k)).fillna(0.0)
        mass_sum = float(mass.sum()) + 1e-12
        for medoid_id in range(k):
            probs_rows.append({
                "class_id": class_id,
                "k": k,
                "half_life_months": hl,
                "beta": float(BETA),
                "medoid_id": int(medoid_id),
                "p_medoid_given_class_weighted": float(mass.iloc[medoid_id] / mass_sum),
                "train_weight_mass": float(mass.iloc[medoid_id]),
            })

        # assign all splits to nearest medoid (unweighted assignment, but medoids are weighted-fit)
        for split in ("train", "validate", "test"):
            sub = extract(df, split, class_id)
            if len(sub) == 0:
                continue
            dates_sub = pd.to_datetime(sub["date"]).to_numpy()
            Z_sub = sub[SHAPE_COLS].to_numpy(dtype=float)
            labels, dmin = assign_to_medoids_by_dot(Z_sub, Z_medoids)

            assign_df = pd.DataFrame({
                "date": [str(pd.Timestamp(d).date()) for d in dates_sub],
                "split": split,
                "class_id": class_id,
                "cluster_id": labels.astype(int),
                "dist_to_medoid": dmin.astype(float),
                "half_life_months": hl,
                "beta": float(BETA),
            })
            assignments_rows.append(assign_df)

            s = split_eval_stats(dmin.astype(float))
            eval_rows.append({
                "class_id": class_id,
                "k": k,
                "half_life_months": hl,
                "beta": float(BETA),
                "split": split,
                "n_days": int(len(assign_df)),
                "mean_dist": s["mean"],
                "p90_dist": s["p90"],
                "p95_dist": s["p95"],
                "max_dist": s["max"],
            })

    medoids_df = pd.DataFrame(medoids_rows)
    probs_df = pd.DataFrame(probs_rows)
    assignments_df = pd.concat(assignments_rows, ignore_index=True) if assignments_rows else pd.DataFrame()
    eval_df = pd.DataFrame(eval_rows)

    out_medoids = out_dir / "B_medoids_shapes.parquet"
    out_probs = out_dir / "B_medoids_probabilities_weighted.parquet"
    out_assign = out_dir / "B_assignments_all_splits.parquet"
    out_eval_csv = out_dir / "B_representation_error_by_split.csv"
    out_eval_pq = out_dir / "B_representation_error_by_split.parquet"

    medoids_df.to_parquet(out_medoids, index=False)
    probs_df.to_parquet(out_probs, index=False)
    assignments_df.to_parquet(out_assign, index=False)
    eval_df.sort_values(["class_id", "split"]).to_csv(out_eval_csv, index=False, encoding="utf-8")
    eval_df.to_parquet(out_eval_pq, index=False)

    # ============================
    # Manifest
    # ============================
    manifest = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "daily_features": str(in_daily),
            "k_by_class": str(in_k),
        },
        "outputs": {
            "drift_csv": str(drift_out),
            "grid_csv": str(grid_out),
            "multistart_stability_csv": str(stability_out),
            "best_half_life_json": str(best_hl_path),
            "B_medoids_shapes": str(out_medoids),
            "B_probabilities_weighted": str(out_probs),
            "B_assignments": str(out_assign),
            "B_representation_error_csv": str(out_eval_csv),
        },
        "params": {
            "beta": float(BETA),
            "half_life_grid_months": HALF_LIFE_GRID_MONTHS,
            "reference_date_mode": REFERENCE_DATE_MODE,
            "reference_date": str(ref_date.date()),
            "n_starts": N_STARTS,
            "pam_max_iter": MAX_ITER,
            "permutations": PERMUTATIONS,
            "drift_pairs": DRIFT_PAIRS,
            "rbf_bandwidth": RBF_BANDWIDTH,
            "validate_selection": "lexicographic(p95,mean,p90,half_life)" if not USE_SCALAR_VALID_SCORE else f"scalar({VALID_SCORE_WEIGHTS})",
        },
        "notes": [
            "Drift tests: Energy Distance and MMD^2 with permutation p-values per class and feature space.",
            "Strategy B: weighted PAM on train with recency weights; half-life chosen using validate-only metrics.",
            "Multi-start: choose best-of-N by weighted train objective; report stability via Jaccard overlap of medoid sets.",
        ],
        "run_dir": str(run_dir),
    }
    (run_dir / "05_manifest_strategy_B.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("=== Script 5 complete (Drift + Grid + Strategy B) ===")
    print(f"Drift tests: {drift_out}")
    print(f"Grid results: {grid_out}")
    print(f"Multi-start stability: {stability_out}")
    print(f"Best half-life per class: {best_hl_path}")
    print(f"B outputs: {out_dir}")
    print(f"Manifest: {run_dir / '05_manifest_strategy_B.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
