from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================

TEST_YEAR = 2025

SHAPE_COLS = [f"shape_z_h{h:02d}" for h in range(24)]
LEVEL_COLS = ["level_mean", "level_max", "level_std"]  # must exist in daily features & medoids
N_HOURS = 24

DO_K_COMMON = True
WRITE_DAY_LEVEL = True


# =========================
# PATHS
# =========================

def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(15):
        if (cur / "1_Configs").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError("Could not locate repo root (folder containing '1_Configs').")


def processed_root(repo_root: Path) -> Path:
    return repo_root / "1_Configs" / "1.2_Data_processed" / "DA-market"


def regimes_dir(repo_root: Path) -> Path:
    return processed_root(repo_root) / "03_regimes"


def reports_dir(repo_root: Path) -> Path:
    out = processed_root(repo_root) / "99_reports"
    out.mkdir(parents=True, exist_ok=True)
    return out


def daily_features_path(repo_root: Path) -> Path:
    p = processed_root(repo_root) / "01_daily_features" / "da_daily_features_scoped_2022_2025.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Daily features parquet not found: {p}")
    return p


def load_parquet_or_csv(path_base: Path) -> pd.DataFrame:
    if path_base.with_suffix(".parquet").exists():
        return pd.read_parquet(path_base.with_suffix(".parquet"))
    if path_base.with_suffix(".csv").exists():
        return pd.read_csv(path_base.with_suffix(".csv"))
    raise FileNotFoundError(f"Expected .parquet or .csv for base: {path_base}")


# =========================
# COLUMN INFERENCE
# =========================

def infer_id_col(df: pd.DataFrame) -> str:
    for c in ["medoid_id", "cluster_id", "prototype_id"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not infer id column. Columns: {list(df.columns)}")


def infer_prob_col(df: pd.DataFrame) -> str:
    for c in ["p_medoid_given_class_pruned", "p_medoid_given_class", "prob", "p", "probability"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not infer probability column. Columns: {list(df.columns)}")


def infer_medoid_id_col(medoids: pd.DataFrame) -> str:
    for c in ["medoid_id", "cluster_id", "prototype_id"]:
        if c in medoids.columns:
            return c
    raise KeyError(f"Could not infer medoid id column in medoids. Columns: {list(medoids.columns)}")


# =========================
# DISTANCES
# =========================

def standardize_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X - mu) / sd


def corr_distance_matrix(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    # 1 - Pearson corr
    Xz = standardize_rows(X.astype(float))
    Mz = standardize_rows(M.astype(float))
    corr = (Xz @ Mz.T) / float(N_HOURS)
    corr = np.clip(corr, -1.0, 1.0)
    return 1.0 - corr


def euclid_distance_matrix(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    # ||x - m||_2
    X = X.astype(float)
    M = M.astype(float)
    # (n, d) and (k, d) => (n, k)
    X2 = np.sum(X * X, axis=1, keepdims=True)
    M2 = np.sum(M * M, axis=1, keepdims=True).T
    # numerical guard
    D2 = np.maximum(X2 + M2 - 2.0 * (X @ M.T), 0.0)
    return np.sqrt(D2)


def zscore_fit(train_like: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = np.mean(train_like, axis=0)
    sd = np.std(train_like, axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return mu, sd


# =========================
# LIBRARY
# =========================

@dataclass(frozen=True)
class Library:
    name: str
    medoids: pd.DataFrame
    probs: pd.DataFrame
    id_col_probs: str
    prob_col: str
    id_col_medoids: str


def load_library(repo_root: Path, strategy: str) -> Library:
    sdir = regimes_dir(repo_root) / f"strategy_{strategy}"
    if not sdir.exists():
        raise FileNotFoundError(f"Strategy folder not found: {sdir}")

    if strategy == "A":
        medoids_path = sdir / "A_medoids_shapes.parquet"
        probs_base = sdir / "A_medoids_probabilities_pruned"
        name = "A_pruned"
    elif strategy == "B":
        medoids_path = sdir / "B_medoids_shapes.parquet"
        probs_base = sdir / "B_medoids_probabilities_pruned"
        name = "B_pruned"
    else:
        raise ValueError("strategy must be 'A' or 'B'")

    if not medoids_path.exists():
        raise FileNotFoundError(f"Missing medoids file: {medoids_path}")

    medoids = pd.read_parquet(medoids_path)
    probs = load_parquet_or_csv(probs_base)

    for c in ["class_id", *SHAPE_COLS, *LEVEL_COLS]:
        if c not in medoids.columns:
            raise KeyError(f"{strategy} medoids missing '{c}'. Columns: {list(medoids.columns)}")
    if "class_id" not in probs.columns:
        raise KeyError(f"{strategy} probs missing 'class_id'. Columns: {list(probs.columns)}")

    id_col_probs = infer_id_col(probs)
    prob_col = infer_prob_col(probs)
    id_col_medoids = infer_medoid_id_col(medoids)

    return Library(
        name=name,
        medoids=medoids.copy(),
        probs=probs.copy(),
        id_col_probs=id_col_probs,
        prob_col=prob_col,
        id_col_medoids=id_col_medoids,
    )


def select_top_k_by_prob(probs: pd.DataFrame, id_col: str, prob_col: str, k: int) -> pd.DataFrame:
    if k <= 0:
        return probs.iloc[0:0].copy()
    return probs.sort_values(prob_col, ascending=False).head(k).copy()


# =========================
# EVAL CORE
# =========================

def eval_one_class(
    days: pd.DataFrame,
    lib: Library,
    class_id: str,
    level_mu: np.ndarray,
    level_sd: np.ndarray,
    k_common: int | None,
) -> Tuple[Dict, pd.DataFrame, Dict]:
    """
    Returns:
      summary_row, day_level_df, calibration_row
    """
    probs_c = lib.probs[lib.probs["class_id"] == class_id].copy()
    if len(probs_c) == 0:
        return (
            {
                "strategy": lib.name,
                "class_id": class_id,
                "n_test_days": int(len(days)),
                "kept_k": 0,
                "note": "no_probs_for_class",
            },
            pd.DataFrame(),
            {
                "strategy": lib.name,
                "class_id": class_id,
                "n_test_days": int(len(days)),
                "kept_k": 0,
                "note": "no_probs_for_class",
            },
        )

    if k_common is not None:
        probs_c = select_top_k_by_prob(probs_c, lib.id_col_probs, lib.prob_col, k_common)

    ids = probs_c[lib.id_col_probs].astype(int).tolist()
    kept_k = int(len(ids))

    # align medoids to probs ids
    med_c = lib.medoids[lib.medoids["class_id"] == class_id].copy()
    med_c[lib.id_col_medoids] = med_c[lib.id_col_medoids].astype(int)
    med_c = med_c[med_c[lib.id_col_medoids].isin(ids)].copy()
    if len(med_c) == 0:
        return (
            {
                "strategy": lib.name,
                "class_id": class_id,
                "n_test_days": int(len(days)),
                "kept_k": kept_k,
                "note": "no_medoids_for_probs_ids",
            },
            pd.DataFrame(),
            {
                "strategy": lib.name,
                "class_id": class_id,
                "n_test_days": int(len(days)),
                "kept_k": kept_k,
                "note": "no_medoids_for_probs_ids",
            },
        )

    med_c = med_c.set_index(lib.id_col_medoids).loc[ids].reset_index()

    # --- shape distances ---
    Xs = days[SHAPE_COLS].to_numpy(dtype=float)
    Ms = med_c[SHAPE_COLS].to_numpy(dtype=float)
    Dshape = corr_distance_matrix(Xs, Ms)

    argmin_shape = np.argmin(Dshape, axis=1)
    dmin_shape = Dshape[np.arange(len(days)), argmin_shape]
    nearest_shape_id = np.array(ids, dtype=int)[argmin_shape]

    p_vec = probs_c[lib.prob_col].to_numpy(dtype=float)
    psum = float(p_vec.sum())
    p_vec = (p_vec / psum) if psum > 0 else (np.ones_like(p_vec) / float(len(p_vec)))
    dexp_shape = Dshape @ p_vec

    # --- level distances (z-scored using TRAIN-level stats) ---
    Xl = days[LEVEL_COLS].to_numpy(dtype=float)
    Ml = med_c[LEVEL_COLS].to_numpy(dtype=float)

    Xlz = (Xl - level_mu) / level_sd
    Mlz = (Ml - level_mu) / level_sd

    Dlevel = euclid_distance_matrix(Xlz, Mlz)
    argmin_level = np.argmin(Dlevel, axis=1)
    dmin_level = Dlevel[np.arange(len(days)), argmin_level]
    nearest_level_id = np.array(ids, dtype=int)[argmin_level]
    dexp_level = Dlevel @ p_vec

    def q(arr: np.ndarray, qq: float) -> float:
        return float(np.quantile(arr, qq))

    summary = {
        "strategy": lib.name,
        "class_id": class_id,
        "n_test_days": int(len(days)),
        "kept_k": kept_k,
        # Shape
        "mean_dmin_shape": float(dmin_shape.mean()),
        "p95_dmin_shape": q(dmin_shape, 0.95),
        "max_dmin_shape": float(dmin_shape.max()),
        "mean_dexp_shape": float(dexp_shape.mean()),
        "p95_dexp_shape": q(dexp_shape, 0.95),
        # Level
        "mean_dmin_level": float(dmin_level.mean()),
        "p95_dmin_level": q(dmin_level, 0.95),
        "max_dmin_level": float(dmin_level.max()),
        "mean_dexp_level": float(dexp_level.mean()),
        "p95_dexp_level": q(dexp_level, 0.95),
        "note": "",
    }

    # --- probability calibration ---
    # empirical frequency of nearest-shape assignments
    emp_counts = pd.Series(nearest_shape_id).value_counts()
    emp = np.array([float(emp_counts.get(mid, 0.0)) for mid in ids], dtype=float)
    emp = emp / float(emp.sum()) if emp.sum() > 0 else np.ones_like(emp) / float(len(emp))

    p = p_vec.copy()

    l1 = float(np.sum(np.abs(emp - p)))
    # Jensen-Shannon divergence (bounded, symmetric)
    m = 0.5 * (emp + p)
    eps = 1e-12

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    js = 0.5 * kl(emp, m) + 0.5 * kl(p, m)

    calib = {
        "strategy": lib.name,
        "class_id": class_id,
        "n_test_days": int(len(days)),
        "kept_k": kept_k,
        "l1_prob_error_nearest_shape": l1,
        "js_div_nearest_shape": js,
        "note": "",
    }

    # day-level output
    day_out = pd.DataFrame(
        {
            "strategy": lib.name,
            "date": pd.to_datetime(days["date"]).dt.date,
            "class_id": class_id,
            "nearest_shape_medoid_id": nearest_shape_id.astype(int),
            "dmin_shape": dmin_shape.astype(float),
            "dexp_shape": dexp_shape.astype(float),
            "nearest_level_medoid_id": nearest_level_id.astype(int),
            "dmin_level": dmin_level.astype(float),
            "dexp_level": dexp_level.astype(float),
        }
    )

    return summary, day_out, calib


def medoid_overlap_report(libA: Library, libB: Library) -> pd.DataFrame:
    """
    Reports overlap by medoid_date (if present) and medoid_id.
    Also reports nearest A->B medoid shape distance summary per class.
    """
    A = libA.medoids.copy()
    B = libB.medoids.copy()

    rows: List[Dict] = []

    has_date_A = "medoid_date" in A.columns
    has_date_B = "medoid_date" in B.columns

    classes = sorted(set(A["class_id"].unique()) | set(B["class_id"].unique()))
    for c in classes:
        Ac = A[A["class_id"] == c].copy()
        Bc = B[B["class_id"] == c].copy()
        if len(Ac) == 0 or len(Bc) == 0:
            rows.append({
                "class_id": c,
                "nA": int(len(Ac)),
                "nB": int(len(Bc)),
                "overlap_medoid_id": 0,
                "overlap_medoid_date": 0 if (has_date_A and has_date_B) else np.nan,
                "mean_nearest_A_to_B_shape_dist": np.nan,
                "p95_nearest_A_to_B_shape_dist": np.nan,
            })
            continue

        idsA = set(Ac[libA.id_col_medoids].astype(int).tolist())
        idsB = set(Bc[libB.id_col_medoids].astype(int).tolist())
        overlap_id = len(idsA.intersection(idsB))

        overlap_date = np.nan
        if has_date_A and has_date_B:
            dA = set(pd.to_datetime(Ac["medoid_date"]).dt.date.tolist())
            dB = set(pd.to_datetime(Bc["medoid_date"]).dt.date.tolist())
            overlap_date = float(len(dA.intersection(dB)))

        # nearest A->B in shape space
        XA = Ac[SHAPE_COLS].to_numpy(dtype=float)
        XB = Bc[SHAPE_COLS].to_numpy(dtype=float)
        D = corr_distance_matrix(XA, XB)
        nearest = np.min(D, axis=1)

        rows.append({
            "class_id": c,
            "nA": int(len(Ac)),
            "nB": int(len(Bc)),
            "overlap_medoid_id": int(overlap_id),
            "overlap_medoid_date": overlap_date,
            "mean_nearest_A_to_B_shape_dist": float(np.mean(nearest)),
            "p95_nearest_A_to_B_shape_dist": float(np.quantile(nearest, 0.95)),
        })

    return pd.DataFrame(rows)


def main() -> int:
    repo_root = find_repo_root(Path(__file__).parent)

    df = pd.read_parquet(daily_features_path(repo_root)).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)

    if "is_complete_24h" in df.columns:
        df = df[df["is_complete_24h"] == True].copy()  # noqa: E712

    test = df[df["year"] == TEST_YEAR].copy()
    if len(test) == 0:
        raise ValueError(f"No test rows for {TEST_YEAR}")

    # Train-level stats for z-scoring LEVEL distances (use 2022-2023 train window)
    train_like = df[df["year"].isin([2022, 2023])].copy()
    level_mu, level_sd = zscore_fit(train_like[LEVEL_COLS].to_numpy(dtype=float))

    libA = load_library(repo_root, "A")
    libB = load_library(repo_root, "B")

    # Determine K-common per class based on pruned probs
    def kept_k_by_class(lib: Library) -> Dict[str, int]:
        return lib.probs.groupby("class_id")[lib.id_col_probs].nunique().to_dict()

    kA = kept_k_by_class(libA)
    kB = kept_k_by_class(libB)
    all_classes = sorted(set(test["class_id"].unique()))
    k_common = {c: int(min(kA.get(c, 0), kB.get(c, 0))) for c in all_classes}

    # Run both modes: as-is and k-common
    modes = [("as_is", None)]
    if DO_K_COMMON:
        modes.append(("k_common", k_common))

    all_summary = []
    all_calib = []
    all_day = []

    for mode_name, kmap in modes:
        summary_rows = []
        calib_rows = []
        day_rows = []

        for class_id in all_classes:
            days = test[test["class_id"] == class_id].copy()
            if len(days) == 0:
                continue

            k_use = None if kmap is None else int(kmap.get(class_id, 0))

            sA, dA, cA = eval_one_class(days, libA, class_id, level_mu, level_sd, k_use)
            sB, dB, cB = eval_one_class(days, libB, class_id, level_mu, level_sd, k_use)

            sA["mode"] = mode_name
            sB["mode"] = mode_name
            cA["mode"] = mode_name
            cB["mode"] = mode_name

            summary_rows.extend([sA, sB])
            calib_rows.extend([cA, cB])

            if WRITE_DAY_LEVEL:
                dA["mode"] = mode_name
                dB["mode"] = mode_name
                day_rows.append(dA)
                day_rows.append(dB)

        summary = pd.DataFrame(summary_rows)
        calib = pd.DataFrame(calib_rows)

        # pivot for A vs B comparison table
        def pivot_pair(metric_cols: List[str]) -> pd.DataFrame:
            keep = ["mode", "class_id", "strategy", *metric_cols]
            sub = summary[keep].copy()
            piv = sub.pivot_table(index=["mode", "class_id"], columns="strategy", values=metric_cols, aggfunc="first")
            piv.columns = [f"{m}_{s}" for m, s in piv.columns]
            return piv.reset_index()

        metric_cols = [
            "n_test_days", "kept_k",
            "mean_dmin_shape", "p95_dmin_shape", "mean_dexp_shape", "p95_dexp_shape",
            "mean_dmin_level", "p95_dmin_level", "mean_dexp_level", "p95_dexp_level",
        ]
        cmp = pivot_pair(metric_cols)

        # add "B better" booleans
        for m in [
            "mean_dmin_shape", "p95_dmin_shape", "mean_dexp_shape", "p95_dexp_shape",
            "mean_dmin_level", "p95_dmin_level", "mean_dexp_level", "p95_dexp_level",
        ]:
            a = cmp.get(f"{m}_A_pruned")
            b = cmp.get(f"{m}_B_pruned")
            if a is not None and b is not None:
                cmp[f"B_better_{m}"] = (b < a)

        out_cmp = reports_dir(repo_root) / f"A_vs_B_pruned_deep_test{TEST_YEAR}_{mode_name}.csv"
        cmp.to_csv(out_cmp, index=False, encoding="utf-8")

        out_cal = reports_dir(repo_root) / f"A_vs_B_probability_calibration_test{TEST_YEAR}_{mode_name}.csv"
        calib.to_csv(out_cal, index=False, encoding="utf-8")

        if WRITE_DAY_LEVEL and len(day_rows) > 0:
            day_all = pd.concat(day_rows, ignore_index=True)
            out_day = reports_dir(repo_root) / f"A_vs_B_day_level_deep_test{TEST_YEAR}_{mode_name}.parquet"
            day_all.to_parquet(out_day, index=False)

        all_summary.append(out_cmp)
        all_calib.append(out_cal)

    # Medoid overlap report (independent of test year)
    overlap = medoid_overlap_report(libA, libB)
    out_overlap = reports_dir(repo_root) / "A_vs_B_medoid_overlap_report.csv"
    overlap.to_csv(out_overlap, index=False, encoding="utf-8")

    print("=== Deep A vs B comparison complete ===")
    for p in all_summary:
        print(f"Wrote: {p}")
    for p in all_calib:
        print(f"Wrote: {p}")
    print(f"Wrote: {out_overlap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
