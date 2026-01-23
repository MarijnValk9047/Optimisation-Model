from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SEED = 12345
N_PERM = 500  # thesis runs: consider 2000+
MIN_GROUP_N = 10

# RBF bandwidth for MMD: "median" recommended
RBF_BANDWIDTH = "median"

# Block permutation methods
PERM_METHODS = ["iid", "week", "biweek", "month"]

# Outputs
OUTPUT_REL = Path("outputs") / "drift_sanity" / "outputs"

# Daily feature columns
SHAPE_COLS = [f"shape_z_h{h:02d}" for h in range(24)]
LEVEL_COLS = ["level_mean", "level_max", "level_std"]


# ============================================================
# UTIL: repo root detection + IO paths
# ============================================================

def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Auto-detect repo root by searching upwards for a folder named '1_Configs'.
    Works from script location (recommended) or current working directory fallback.
    """
    candidates = []
    if start is not None:
        candidates.append(start.resolve())
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        # __file__ may not exist in some interactive contexts
        pass
    candidates.append(Path.cwd().resolve())

    for base in candidates:
        cur = base if base.is_dir() else base.parent
        for _ in range(20):  # safety
            if (cur / "1_Configs").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent

    raise FileNotFoundError("Could not locate repo root: no '1_Configs' found in parent paths.")


def default_daily_features_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "1_Configs"
        / "1.2_Data_processed"
        / "DA-market"
        / "01_daily_features"
        / "da_daily_features_scoped_2022_2025.parquet"
    )


# ============================================================
# STAT TESTS: Energy distance + MMD^2 + permutation p-values
# ============================================================

def _mean_offdiag(A: np.ndarray) -> float:
    n = A.shape[0]
    if n <= 1:
        return 0.0
    return float((A.sum() - np.trace(A)) / (n * (n - 1)))


def _pairwise_euclidean(X: np.ndarray) -> np.ndarray:
    X2 = np.sum(X * X, axis=1, keepdims=True)
    dist2 = X2 + X2.T - 2.0 * (X @ X.T)
    np.maximum(dist2, 0.0, out=dist2)
    return np.sqrt(dist2)


def energy_stat_from_dist(D: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    d_ab = D[np.ix_(idx_a, idx_b)]
    e_ab = float(d_ab.mean()) if d_ab.size else 0.0
    e_aa = _mean_offdiag(D[np.ix_(idx_a, idx_a)])
    e_bb = _mean_offdiag(D[np.ix_(idx_b, idx_b)])
    return float(2.0 * e_ab - e_aa - e_bb)


def rbf_kernel_matrix(X: np.ndarray, gamma: float) -> np.ndarray:
    X2 = np.sum(X * X, axis=1, keepdims=True)
    dist2 = X2 + X2.T - 2.0 * (X @ X.T)
    np.maximum(dist2, 0.0, out=dist2)
    return np.exp(-gamma * dist2)


def mmd2_stat_from_kernel(K: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    K_aa = K[np.ix_(idx_a, idx_a)]
    K_bb = K[np.ix_(idx_b, idx_b)]
    K_ab = K[np.ix_(idx_a, idx_b)]
    m_aa = _mean_offdiag(K_aa)
    m_bb = _mean_offdiag(K_bb)
    m_ab = float(K_ab.mean()) if K_ab.size else 0.0
    return float(m_aa + m_bb - 2.0 * m_ab)


def permutation_pvalue(stat_obs: float, stats_perm: np.ndarray) -> float:
    return float((1.0 + np.sum(stats_perm >= stat_obs)) / (1.0 + len(stats_perm)))


@dataclass(frozen=True)
class DriftResult:
    class_id: str
    comparison: str
    placebo_type: str
    feature_space: str
    nA: int
    nB: int
    stat_energy: float
    p_energy: float
    stat_mmd2: float
    p_mmd2: float
    n_perm: int
    seed: int
    perm_method: str


# ============================================================
# BLOCKING / PERMUTATION SCHEMES
# ============================================================

def make_block_ids(dates: pd.Series, perm_method: str) -> pd.Series:
    d = pd.to_datetime(dates)
    if perm_method == "iid":
        # each row its own block
        return pd.Series(np.arange(len(d)), index=d.index).astype(str)

    if perm_method == "week":
        iso = d.dt.isocalendar()
        return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).astype(str)

    if perm_method == "biweek":
        iso = d.dt.isocalendar()
        bi = ((iso["week"] - 1) // 2) + 1
        return (iso["year"].astype(str) + "-BW" + bi.astype(str).str.zfill(2)).astype(str)

    if perm_method == "month":
        return (d.dt.year.astype(str) + "-" + d.dt.month.astype(str).str.zfill(2)).astype(str)

    raise ValueError(f"Unknown perm_method: {perm_method}")


def block_permute_indices(
    block_ids: np.ndarray,
    nA_target: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Permute assignment at block level: shuffle blocks, then assign blocks to group A
    until we reach >= nA_target; remaining blocks go to B.

    This keeps time dependence within blocks and keeps group sizes approximately fixed.
    """
    unique_blocks, inv = np.unique(block_ids, return_inverse=True)
    rng.shuffle(unique_blocks)

    # map block -> indices
    block_to_idx: Dict[str, np.ndarray] = {}
    for b in unique_blocks:
        block_to_idx[str(b)] = np.where(block_ids == b)[0]

    A_idx: List[int] = []
    total = 0
    for b in unique_blocks:
        idxs = block_to_idx[str(b)]
        A_idx.extend(idxs.tolist())
        total += len(idxs)
        if total >= nA_target:
            break

    A_idx_arr = np.array(sorted(set(A_idx)), dtype=int)
    all_idx = np.arange(len(block_ids), dtype=int)
    mask = np.ones(len(block_ids), dtype=bool)
    mask[A_idx_arr] = False
    B_idx_arr = all_idx[mask]
    return A_idx_arr, B_idx_arr


def iid_permute_indices(
    n: int,
    nA_target: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(np.arange(n, dtype=int))
    A = perm[:nA_target]
    B = perm[nA_target:]
    return np.array(A, dtype=int), np.array(B, dtype=int)


# ============================================================
# CORE DRIFT TEST RUNNER
# ============================================================

def run_drift_test(
    X: np.ndarray,
    idxA: np.ndarray,
    idxB: np.ndarray,
    dates: pd.Series,
    perm_method: str,
    n_perm: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    """
    Returns: (energy_stat, p_energy, mmd2_stat, p_mmd2)
    Permutation test shuffles labels under the null.
    """
    rng = np.random.default_rng(seed)

    nA = len(idxA)
    nB = len(idxB)
    if nA < MIN_GROUP_N or nB < MIN_GROUP_N:
        return (np.nan, np.nan, np.nan, np.nan)

    # Precompute distances for energy
    D = _pairwise_euclidean(X)
    energy_obs = energy_stat_from_dist(D, idxA, idxB)

    # MMD: bandwidth
    if RBF_BANDWIDTH == "median":
        tri = D[np.triu_indices(D.shape[0], k=1)]
        med = float(np.median(tri)) if tri.size else 1.0
        sigma = med if med > 1e-12 else 1.0
    else:
        sigma = float(RBF_BANDWIDTH)
    gamma = 1.0 / (2.0 * sigma * sigma)
    K = rbf_kernel_matrix(X, gamma)
    mmd_obs = mmd2_stat_from_kernel(K, idxA, idxB)

    # Permutations
    e_perm = np.empty(n_perm, dtype=float)
    m_perm = np.empty(n_perm, dtype=float)

    block_ids = make_block_ids(dates, perm_method).to_numpy()

    for p in range(n_perm):
        if perm_method == "iid":
            pA, pB = iid_permute_indices(n=len(X), nA_target=nA, rng=rng)
        else:
            pA, pB = block_permute_indices(block_ids=block_ids, nA_target=nA, rng=rng)

        # ensure min size (block assignment can overshoot A size; B can become too small)
        if len(pA) < MIN_GROUP_N or len(pB) < MIN_GROUP_N:
            e_perm[p] = np.nan
            m_perm[p] = np.nan
            continue

        e_perm[p] = energy_stat_from_dist(D, pA, pB)
        m_perm[p] = mmd2_stat_from_kernel(K, pA, pB)

    e_perm = e_perm[~np.isnan(e_perm)]
    m_perm = m_perm[~np.isnan(m_perm)]
    if len(e_perm) == 0 or len(m_perm) == 0:
        return (energy_obs, np.nan, mmd_obs, np.nan)

    p_energy = permutation_pvalue(energy_obs, e_perm)
    p_mmd2 = permutation_pvalue(mmd_obs, m_perm)
    return (energy_obs, p_energy, mmd_obs, p_mmd2)


# ============================================================
# DATA HELPERS
# ============================================================

def load_daily_features(repo_root: Path) -> pd.DataFrame:
    p = default_daily_features_path(repo_root)
    if not p.exists():
        raise FileNotFoundError(f"Daily features parquet not found: {p}")
    df = pd.read_parquet(p)
    if "date" not in df.columns:
        raise ValueError("Daily features missing required column: date")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)

    if "is_complete_24h" in df.columns:
        df = df[df["is_complete_24h"] == True].copy()  # noqa: E712

    # ensure required cols exist
    missing = [c for c in SHAPE_COLS + LEVEL_COLS + ["class_id"] if c not in df.columns]
    if missing:
        raise ValueError(f"Daily features missing columns: {missing}")

    # drop NAs in required spaces (we will also do per-task filtering)
    return df


def subset_class_year(df: pd.DataFrame, class_id: str, years: Iterable[int]) -> pd.DataFrame:
    sub = df[(df["class_id"] == class_id) & (df["year"].isin(list(years)))].copy()
    sub = sub.sort_values("date")
    return sub


# ============================================================
# (1) PLACEBO CHECKS
# ============================================================

def run_placebo_checks(df: pd.DataFrame, seed: int, n_perm: int) -> List[DriftResult]:
    results: List[DriftResult] = []
    rng = np.random.default_rng(seed)

    classes = sorted(df["class_id"].unique())

    # Placebo A: 2022 vs 2023 (NOT true placebo for levels; label accordingly)
    placebo_a_type = "placebo_A_year_2022_vs_2023"
    for class_id in classes:
        sub = subset_class_year(df, class_id, years=[2022, 2023])
        A = sub[sub["year"] == 2022]
        B = sub[sub["year"] == 2023]
        if len(A) < MIN_GROUP_N or len(B) < MIN_GROUP_N:
            print(f"[WARN] {placebo_a_type} skip {class_id}: nA={len(A)}, nB={len(B)}")
            continue

        idxA = np.arange(len(A), dtype=int)
        idxB = np.arange(len(A), len(A) + len(B), dtype=int)
        pooled = pd.concat([A, B], ignore_index=True)
        dates = pooled["date"]

        for feature_space, cols in [("shape_24D", SHAPE_COLS), ("level_3D", LEVEL_COLS)]:
            X = pooled[cols].to_numpy(dtype=float)
            e, p_e, m, p_m = run_drift_test(
                X=X, idxA=idxA, idxB=idxB, dates=dates,
                perm_method="iid", n_perm=n_perm, seed=seed + 100 + hash((class_id, placebo_a_type, feature_space)) % 10_000
            )
            results.append(DriftResult(
                class_id=class_id,
                comparison="2022_vs_2023",
                placebo_type=placebo_a_type,
                feature_space=feature_space,
                nA=len(idxA),
                nB=len(idxB),
                stat_energy=e,
                p_energy=p_e,
                stat_mmd2=m,
                p_mmd2=p_m,
                n_perm=n_perm,
                seed=seed,
                perm_method="iid",
            ))

    # Placebo B: within-year random halves for 2024 and 2025 (true placebo)
    for year in [2024, 2025]:
        placebo_b_type = f"placebo_B_random_halves_{year}"
        for class_id in classes:
            sub = subset_class_year(df, class_id, years=[year])
            n = len(sub)
            if n < 2 * MIN_GROUP_N:
                print(f"[WARN] {placebo_b_type} skip {class_id}: n={n}")
                continue

            # deterministic split
            idx = np.arange(n, dtype=int)
            rng_local = np.random.default_rng(seed + 2000 + year * 10 + (abs(hash(class_id)) % 10_000))
            rng_local.shuffle(idx)
            nA = n // 2
            A_idx = np.sort(idx[:nA])
            B_idx = np.sort(idx[nA:])

            dates = sub["date"]
            for feature_space, cols in [("shape_24D", SHAPE_COLS), ("level_3D", LEVEL_COLS)]:
                X = sub[cols].to_numpy(dtype=float)
                e, p_e, m, p_m = run_drift_test(
                    X=X, idxA=A_idx, idxB=B_idx, dates=dates,
                    perm_method="iid", n_perm=n_perm, seed=seed + 300 + hash((class_id, placebo_b_type, feature_space)) % 10_000
                )
                results.append(DriftResult(
                    class_id=class_id,
                    comparison=f"{year}_random_half_vs_half",
                    placebo_type=placebo_b_type,
                    feature_space=feature_space,
                    nA=len(A_idx),
                    nB=len(B_idx),
                    stat_energy=e,
                    p_energy=p_e,
                    stat_mmd2=m,
                    p_mmd2=p_m,
                    n_perm=n_perm,
                    seed=seed,
                    perm_method="iid",
                ))

    # Placebo C: within-year time split Jan–Jun vs Jul–Dec for 2024 and 2025 (true placebo)
    for year in [2024, 2025]:
        placebo_c_type = f"placebo_C_time_halves_{year}"
        for class_id in classes:
            sub = subset_class_year(df, class_id, years=[year])
            if len(sub) == 0:
                continue

            A = sub[sub["date"].dt.month <= 6].copy()
            B = sub[sub["date"].dt.month >= 7].copy()
            if len(A) < MIN_GROUP_N or len(B) < MIN_GROUP_N:
                print(f"[WARN] {placebo_c_type} skip {class_id}: nA={len(A)}, nB={len(B)}")
                continue

            pooled = pd.concat([A, B], ignore_index=True)
            idxA = np.arange(len(A), dtype=int)
            idxB = np.arange(len(A), len(A) + len(B), dtype=int)
            dates = pooled["date"]

            for feature_space, cols in [("shape_24D", SHAPE_COLS), ("level_3D", LEVEL_COLS)]:
                X = pooled[cols].to_numpy(dtype=float)
                e, p_e, m, p_m = run_drift_test(
                    X=X, idxA=idxA, idxB=idxB, dates=dates,
                    perm_method="iid", n_perm=n_perm, seed=seed + 400 + hash((class_id, placebo_c_type, feature_space)) % 10_000
                )
                results.append(DriftResult(
                    class_id=class_id,
                    comparison=f"{year}_time_half_vs_half",
                    placebo_type=placebo_c_type,
                    feature_space=feature_space,
                    nA=len(idxA),
                    nB=len(idxB),
                    stat_energy=e,
                    p_energy=p_e,
                    stat_mmd2=m,
                    p_mmd2=p_m,
                    n_perm=n_perm,
                    seed=seed,
                    perm_method="iid",
                ))

    return results


def placebo_summary_table(df_res: pd.DataFrame) -> pd.DataFrame:
    """
    For true placebos (B and C), compute fraction significant at p<0.05 and p<0.01,
    split by feature space and by statistic type.
    """
    rows = []
    true_placebos = df_res[df_res["placebo_type"].str.startswith("placebo_B") | df_res["placebo_type"].str.startswith("placebo_C")].copy()

    for placebo_type in sorted(true_placebos["placebo_type"].unique()):
        sub = true_placebos[true_placebos["placebo_type"] == placebo_type]
        for feature_space in sorted(sub["feature_space"].unique()):
            ss = sub[sub["feature_space"] == feature_space]

            for stat_name in ["p_energy", "p_mmd2"]:
                pvals = ss[stat_name].dropna().to_numpy(dtype=float)
                if pvals.size == 0:
                    continue
                rows.append({
                    "placebo_type": placebo_type,
                    "feature_space": feature_space,
                    "stat": stat_name,
                    "n_tests": int(pvals.size),
                    "frac_p_lt_0p05": float(np.mean(pvals < 0.05)),
                    "frac_p_lt_0p01": float(np.mean(pvals < 0.01)),
                })

    return pd.DataFrame(rows)


# ============================================================
# (2) YEAR-BY-YEAR DRIFT (iid)
# ============================================================

def run_year_by_year(df: pd.DataFrame, seed: int, n_perm: int) -> List[DriftResult]:
    results: List[DriftResult] = []
    classes = sorted(df["class_id"].unique())
    comparisons = [(2022, 2023), (2023, 2024), (2024, 2025)]

    for yA, yB in comparisons:
        comp_name = f"{yA}_vs_{yB}"
        placebo_type = "year_by_year"
        for class_id in classes:
            A = subset_class_year(df, class_id, years=[yA])
            B = subset_class_year(df, class_id, years=[yB])
            if len(A) < MIN_GROUP_N or len(B) < MIN_GROUP_N:
                print(f"[WARN] year_by_year skip {class_id} {comp_name}: nA={len(A)}, nB={len(B)}")
                continue

            pooled = pd.concat([A, B], ignore_index=True)
            idxA = np.arange(len(A), dtype=int)
            idxB = np.arange(len(A), len(A) + len(B), dtype=int)
            dates = pooled["date"]

            for feature_space, cols in [("shape_24D", SHAPE_COLS), ("level_3D", LEVEL_COLS)]:
                X = pooled[cols].to_numpy(dtype=float)
                e, p_e, m, p_m = run_drift_test(
                    X=X, idxA=idxA, idxB=idxB, dates=dates,
                    perm_method="iid", n_perm=n_perm, seed=seed + 500 + hash((class_id, comp_name, feature_space)) % 10_000
                )
                results.append(DriftResult(
                    class_id=class_id,
                    comparison=comp_name,
                    placebo_type=placebo_type,
                    feature_space=feature_space,
                    nA=len(idxA),
                    nB=len(idxB),
                    stat_energy=e,
                    p_energy=p_e,
                    stat_mmd2=m,
                    p_mmd2=p_m,
                    n_perm=n_perm,
                    seed=seed,
                    perm_method="iid",
                ))

    return results


# ============================================================
# (3) BLOCK PERMUTATION CHECKS (dependence-aware)
# ============================================================

def run_blockperm_checks(df: pd.DataFrame, seed: int, n_perm: int) -> List[DriftResult]:
    results: List[DriftResult] = []
    classes = sorted(df["class_id"].unique())
    comparisons = [(2022, 2023), (2023, 2024), (2024, 2025)]

    for yA, yB in comparisons:
        comp_name = f"{yA}_vs_{yB}"
        placebo_type = "blockperm_year_by_year"
        for class_id in classes:
            A = subset_class_year(df, class_id, years=[yA])
            B = subset_class_year(df, class_id, years=[yB])
            if len(A) < MIN_GROUP_N or len(B) < MIN_GROUP_N:
                print(f"[WARN] blockperm skip {class_id} {comp_name}: nA={len(A)}, nB={len(B)}")
                continue

            pooled = pd.concat([A, B], ignore_index=True)
            idxA = np.arange(len(A), dtype=int)
            idxB = np.arange(len(A), len(A) + len(B), dtype=int)
            dates = pooled["date"]

            for feature_space, cols in [("shape_24D", SHAPE_COLS), ("level_3D", LEVEL_COLS)]:
                X = pooled[cols].to_numpy(dtype=float)

                for perm_method in PERM_METHODS:
                    # For "iid" here, this replicates year_by_year iid, but helps uniform plotting
                    e, p_e, m, p_m = run_drift_test(
                        X=X, idxA=idxA, idxB=idxB, dates=dates,
                        perm_method=perm_method, n_perm=n_perm,
                        seed=seed + 600 + hash((class_id, comp_name, feature_space, perm_method)) % 10_000
                    )
                    results.append(DriftResult(
                        class_id=class_id,
                        comparison=comp_name,
                        placebo_type=placebo_type,
                        feature_space=feature_space,
                        nA=len(idxA),
                        nB=len(idxB),
                        stat_energy=e,
                        p_energy=p_e,
                        stat_mmd2=m,
                        p_mmd2=p_m,
                        n_perm=n_perm,
                        seed=seed,
                        perm_method=perm_method,
                    ))

    return results


def flag_blockperm_inconsistencies(df_block: pd.DataFrame) -> pd.DataFrame:
    """
    Flag cases where iid p < 0.01 but biweek p > 0.05 or month p > 0.05
    for same (class, comparison, feature_space, stat).
    """
    rows = []

    key_cols = ["class_id", "comparison", "feature_space"]
    for stat in ["p_energy", "p_mmd2"]:
        pivot = df_block.pivot_table(
            index=key_cols,
            columns="perm_method",
            values=stat,
            aggfunc="first",
        ).reset_index()

        if "iid" not in pivot.columns:
            continue

        for _, r in pivot.iterrows():
            p_iid = r.get("iid", np.nan)
            p_bi = r.get("biweek", np.nan)
            p_mo = r.get("month", np.nan)

            if np.isnan(p_iid):
                continue

            flag = False
            reason = []
            if p_iid < 0.01:
                if not np.isnan(p_bi) and p_bi > 0.05:
                    flag = True
                    reason.append("iid<0.01 but biweek>0.05")
                if not np.isnan(p_mo) and p_mo > 0.05:
                    flag = True
                    reason.append("iid<0.01 but month>0.05")

            if flag:
                rows.append({
                    "class_id": r["class_id"],
                    "comparison": r["comparison"],
                    "feature_space": r["feature_space"],
                    "stat": stat,
                    "p_iid": float(p_iid),
                    "p_biweek": float(p_bi) if not np.isnan(p_bi) else np.nan,
                    "p_month": float(p_mo) if not np.isnan(p_mo) else np.nan,
                    "flag_reason": "; ".join(reason),
                })

    return pd.DataFrame(rows)


# ============================================================
# SAVING + PLOTTING
# ============================================================

def save_results(repo_root: Path, placebo: List[DriftResult], yby: List[DriftResult], block: List[DriftResult]) -> Tuple[Path, Path, Path]:
    out_dir = repo_root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)

    def to_df(xs: List[DriftResult]) -> pd.DataFrame:
        return pd.DataFrame([x.__dict__ for x in xs])

    p_placebo = out_dir / "drift_placebo_results.csv"
    p_yby = out_dir / "drift_year_by_year_results.csv"
    p_block = out_dir / "drift_blockperm_results.csv"

    to_df(placebo).to_csv(p_placebo, index=False, encoding="utf-8")
    to_df(yby).to_csv(p_yby, index=False, encoding="utf-8")
    to_df(block).to_csv(p_block, index=False, encoding="utf-8")

    return p_placebo, p_yby, p_block


def _heatmap(
    ax: plt.Axes,
    mat: np.ndarray,
    xlabels: List[str],
    ylabels: List[str],
    title: str,
) -> None:
    # -log10(p) heatmap; clip to avoid inf
    safe = np.clip(mat, 1e-12, 1.0)
    z = -np.log10(safe)
    im = ax.imshow(z, aspect="auto")
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="-log10(p)")


def plot_heatmaps_year_by_year(repo_root: Path, df_yby: pd.DataFrame) -> List[Path]:
    out_dir = repo_root / OUTPUT_REL
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    comparisons = sorted(df_yby["comparison"].unique())
    classes = sorted(df_yby["class_id"].unique())

    for feature_space in sorted(df_yby["feature_space"].unique()):
        for stat in ["p_energy", "p_mmd2"]:
            mat = np.full((len(classes), len(comparisons)), np.nan, dtype=float)
            for i, c in enumerate(classes):
                for j, comp in enumerate(comparisons):
                    sub = df_yby[
                        (df_yby["class_id"] == c) &
                        (df_yby["comparison"] == comp) &
                        (df_yby["feature_space"] == feature_space)
                    ]
                    if len(sub) == 0:
                        continue
                    mat[i, j] = float(sub.iloc[0][stat])

            fig, ax = plt.subplots(figsize=(10, 4))
            _heatmap(ax, mat, comparisons, classes, title=f"Year-by-year drift ({feature_space}) — {stat}")
            fig.tight_layout()
            p = plot_dir / f"heatmap_year_by_year_{feature_space}_{stat}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            paths.append(p)

    return paths


def plot_heatmaps_blockperm(repo_root: Path, df_block: pd.DataFrame) -> List[Path]:
    out_dir = repo_root / OUTPUT_REL
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []
    comparisons = sorted(df_block["comparison"].unique())
    classes = sorted(df_block["class_id"].unique())

    for perm_method in PERM_METHODS:
        sub_pm = df_block[df_block["perm_method"] == perm_method].copy()
        for feature_space in sorted(sub_pm["feature_space"].unique()):
            for stat in ["p_energy", "p_mmd2"]:
                mat = np.full((len(classes), len(comparisons)), np.nan, dtype=float)
                for i, c in enumerate(classes):
                    for j, comp in enumerate(comparisons):
                        sub = sub_pm[
                            (sub_pm["class_id"] == c) &
                            (sub_pm["comparison"] == comp) &
                            (sub_pm["feature_space"] == feature_space)
                        ]
                        if len(sub) == 0:
                            continue
                        mat[i, j] = float(sub.iloc[0][stat])

                fig, ax = plt.subplots(figsize=(10, 4))
                _heatmap(ax, mat, comparisons, classes, title=f"Blockperm drift ({perm_method}) — {feature_space} — {stat}")
                fig.tight_layout()
                p = plot_dir / f"heatmap_blockperm_{perm_method}_{feature_space}_{stat}.png"
                fig.savefig(p, dpi=200)
                plt.close(fig)
                paths.append(p)

    return paths


def plot_significance_bars(repo_root: Path, df_yby: pd.DataFrame, df_placebo: pd.DataFrame) -> List[Path]:
    out_dir = repo_root / OUTPUT_REL
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path] = []

    # Year-by-year bar plots: frac significant per comparison
    for feature_space in sorted(df_yby["feature_space"].unique()):
        for stat in ["p_energy", "p_mmd2"]:
            comps = sorted(df_yby["comparison"].unique())
            frac_005 = []
            frac_001 = []
            for comp in comps:
                sub = df_yby[(df_yby["comparison"] == comp) & (df_yby["feature_space"] == feature_space)][stat].dropna().to_numpy()
                if sub.size == 0:
                    frac_005.append(np.nan)
                    frac_001.append(np.nan)
                else:
                    frac_005.append(float(np.mean(sub < 0.05)))
                    frac_001.append(float(np.mean(sub < 0.01)))

            x = np.arange(len(comps))
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(x - 0.15, frac_005, width=0.3, label="p<0.05")
            ax.bar(x + 0.15, frac_001, width=0.3, label="p<0.01")
            ax.set_xticks(x)
            ax.set_xticklabels(comps, rotation=30, ha="right")
            ax.set_ylim(0, 1)
            ax.set_title(f"Year-by-year significant fraction — {feature_space} — {stat}")
            ax.legend()
            fig.tight_layout()
            p = plot_dir / f"bar_year_by_year_{feature_space}_{stat}.png"
            fig.savefig(p, dpi=200)
            plt.close(fig)
            paths.append(p)

    # Placebo B and C separately
    true_placebos = df_placebo[
        df_placebo["placebo_type"].str.startswith("placebo_B") |
        df_placebo["placebo_type"].str.startswith("placebo_C")
    ].copy()

    for placebo_prefix in ["placebo_B", "placebo_C"]:
        sub_p = true_placebos[true_placebos["placebo_type"].str.startswith(placebo_prefix)].copy()
        if len(sub_p) == 0:
            continue

        placebo_types = sorted(sub_p["placebo_type"].unique())
        for feature_space in sorted(sub_p["feature_space"].unique()):
            for stat in ["p_energy", "p_mmd2"]:
                frac_005 = []
                frac_001 = []
                for pt in placebo_types:
                    vals = sub_p[(sub_p["placebo_type"] == pt) & (sub_p["feature_space"] == feature_space)][stat].dropna().to_numpy()
                    if vals.size == 0:
                        frac_005.append(np.nan)
                        frac_001.append(np.nan)
                    else:
                        frac_005.append(float(np.mean(vals < 0.05)))
                        frac_001.append(float(np.mean(vals < 0.01)))

                x = np.arange(len(placebo_types))
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar(x - 0.15, frac_005, width=0.3, label="p<0.05")
                ax.bar(x + 0.15, frac_001, width=0.3, label="p<0.01")
                ax.set_xticks(x)
                ax.set_xticklabels(placebo_types, rotation=30, ha="right")
                ax.set_ylim(0, 1)
                ax.set_title(f"{placebo_prefix}: significant fraction — {feature_space} — {stat}")
                ax.legend()
                fig.tight_layout()
                p = plot_dir / f"bar_{placebo_prefix}_{feature_space}_{stat}.png"
                fig.savefig(p, dpi=200)
                plt.close(fig)
                paths.append(p)

    return paths


# ============================================================
# SUMMARY / INTERPRETATION
# ============================================================

def write_summary_md(
    repo_root: Path,
    df_placebo: pd.DataFrame,
    df_yby: pd.DataFrame,
    df_block: pd.DataFrame,
    placebo_summary: pd.DataFrame,
    flags: pd.DataFrame,
) -> Path:
    out_dir = repo_root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "summary.md"

    lines: List[str] = []
    lines.append("# Drift sanity checks summary\n")

    # Placebos
    lines.append("## Placebo modes\n")
    lines.append("**Placebo A (2022 vs 2023):** kept for context (early shift inside train).")
    lines.append("Not a *true* placebo for **level_3D** because price levels genuinely changed between 2022 and 2023.\n")
    lines.append("**Placebo B (true placebo):** within-year random halves (2024 and 2025).")
    lines.append("Under the null, significant fractions should be near ~5% (p<0.05) and ~1% (p<0.01).\n")
    lines.append("**Placebo C (true placebo):** within-year time split Jan–Jun vs Jul–Dec (2024 and 2025).")
    lines.append("This is stricter and probes mild temporal dependence; it can naturally be a bit more sensitive.\n")

    lines.append("## True placebo summary (B/C only)\n")
    if len(placebo_summary) == 0:
        lines.append("_No true placebo results available (likely due to small sample sizes)._")
    else:
        lines.append(placebo_summary.to_markdown(index=False))

    # Red-flag heuristics
    lines.append("\n## Red-flag heuristics\n")
    lines.append("If Placebo B/C shows very high significant fractions (e.g., >20% at p<0.01), that can indicate:")
    lines.append("- time dependence not handled by iid permutation,")
    lines.append("- preprocessing mismatch between groups,")
    lines.append("- DST alignment / missing-hour artifacts,")
    lines.append("- or a test that’s too sensitive for your dependence structure.\n")

    # Year-by-year drift directionality
    lines.append("## Year-by-year drift (iid permutation)\n")
    if len(df_yby) == 0:
        lines.append("_No year-by-year results._")
    else:
        # compact significance view
        view = df_yby.copy()
        for stat in ["p_energy", "p_mmd2"]:
            view[f"{stat}_sig_0.01"] = view[stat] < 0.01
            view[f"{stat}_sig_0.05"] = view[stat] < 0.05
        keep = ["class_id", "comparison", "feature_space", "p_energy", "p_mmd2", "p_energy_sig_0.01", "p_mmd2_sig_0.01"]
        lines.append(view[keep].sort_values(["feature_space", "comparison", "class_id"]).to_markdown(index=False))

    # Block permutation flags
    lines.append("\n## Block permutation sensitivity\n")
    lines.append("We flag cases where **iid** reports strong significance (p<0.01) but **biweek** or **month** loses significance (p>0.05).")
    if len(flags) == 0:
        lines.append("_No flagged inconsistencies under the given rule._")
    else:
        lines.append(flags.to_markdown(index=False))

    # Interpretation guide
    lines.append("\n## Interpretation guide\n")
    lines.append("### How to read Placebo A vs true placebo B/C\n")
    lines.append("- **Placebo A (2022 vs 2023)** is useful context for *within-train* changes, but it is **not a null** for level drift.")
    lines.append("- **Placebo B/C** are the actual checks of false positive rate under (approximately) stationary conditions.\n")

    lines.append("### What placebo red flags mean\n")
    lines.append("If true placebos (B/C) show too many significant results, likely causes include:")
    lines.append("- pipeline artifacts (different preprocessing between subsets),")
    lines.append("- time alignment/DST errors causing systematic shape differences,")
    lines.append("- dependence (days not i.i.d.) making iid permutation too optimistic,")
    lines.append("- or hidden conditioning leakage.\n")

    lines.append("### What to do next if you see red flags\n")
    lines.append("- Increase permutations to 2000 for more stable p-values (and a smaller p-value floor).")
    lines.append("- Prefer **block permutations** (biweek/month) for inference if dependence is strong.")
    lines.append("- Add a negative-control drift test within a single year using multiple random splits.")
    lines.append("- Verify DST handling: missing hours, duplicated hours, day boundary alignment.")
    lines.append("- Check if 2025 downsampling or timestamp parsing differs from earlier years.\n")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def print_console_checklist(
    p_placebo: Path,
    p_yby: Path,
    p_block: Path,
    p_summary: Path,
    plot_paths: List[Path],
) -> None:
    print("\n=== Drift sanity outputs ===")
    print(f"[OK] Placebo results:     {p_placebo}")
    print(f"[OK] Year-by-year:       {p_yby}")
    print(f"[OK] Block permutation:  {p_block}")
    print(f"[OK] Summary:            {p_summary}")
    if plot_paths:
        print("[OK] Plots:")
        for p in plot_paths:
            print(f"     - {p}")
    print("\nChecklist:")
    print("  - Verify Placebo B/C significant fractions ~5% at p<0.05 and ~1% at p<0.01.")
    print("  - If Placebo C is higher: interpret as temporal dependence sensitivity.")
    print("  - Check flagged iid-vs-block inconsistencies (biweek/month).")
    print("  - Use year-by-year to narrate gradual vs abrupt drift.\n")


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    repo_root = find_repo_root()
    df = load_daily_features(repo_root)

    # Drop any rows missing required columns (per space later)
    df = df.dropna(subset=["class_id", "date"]).copy()

    # Run checks
    print("Running (1) placebo checks ...")
    placebo_results = run_placebo_checks(df, seed=SEED, n_perm=N_PERM)

    print("Running (2) year-by-year drift (iid) ...")
    yby_results = run_year_by_year(df, seed=SEED, n_perm=N_PERM)

    print("Running (3) block permutation checks ...")
    block_results = run_blockperm_checks(df, seed=SEED, n_perm=N_PERM)

    # Save
    p_placebo, p_yby, p_block = save_results(repo_root, placebo_results, yby_results, block_results)

    df_placebo = pd.read_csv(p_placebo)
    df_yby = pd.read_csv(p_yby)
    df_block = pd.read_csv(p_block)

    # Summaries
    placebo_sum = placebo_summary_table(df_placebo)

    # Red-flag heuristic for placebos (B/C)
    red_flags = placebo_sum[
        (placebo_sum["frac_p_lt_0p01"] > 0.20)
    ].copy()

    if len(red_flags) > 0:
        print("\n[RED FLAG] True placebo significant fraction > 20% at p<0.01 detected:")
        print(red_flags.to_string(index=False))
        print("Likely causes: dependence, preprocessing mismatch, DST/time alignment, or overly sensitive test.\n")

    flags_block = flag_blockperm_inconsistencies(df_block)
    if len(flags_block) > 0:
        print("\n[FLAG] iid significant but blockperm not significant (biweek/month) in some cases:")
        print(flags_block.to_string(index=False))
        print()

    # Plots
    print("Plotting heatmaps and bar plots ...")
    plot_paths: List[Path] = []
    if len(df_yby) > 0:
        plot_paths.extend(plot_heatmaps_year_by_year(repo_root, df_yby))
    if len(df_block) > 0:
        plot_paths.extend(plot_heatmaps_blockperm(repo_root, df_block))
    if len(df_yby) > 0 and len(df_placebo) > 0:
        plot_paths.extend(plot_significance_bars(repo_root, df_yby, df_placebo))

    # Summary.md
    p_summary = write_summary_md(
        repo_root=repo_root,
        df_placebo=df_placebo,
        df_yby=df_yby,
        df_block=df_block,
        placebo_summary=placebo_sum,
        flags=flags_block,
    )

    # Console checklist
    print_console_checklist(p_placebo, p_yby, p_block, p_summary, plot_paths)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
