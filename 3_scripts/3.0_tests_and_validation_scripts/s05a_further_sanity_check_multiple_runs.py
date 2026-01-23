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

# Permutations per drift test (p-value resolution is 1/(N_PERM+1))
N_PERM = 500  # for final reporting, consider 2000+

# Repeats for placebo random halves (true placebo calibration)
N_RANDOM_SPLITS_PER_YEAR = 100  # 50–200 is usually enough to see if FPR is sane

MIN_GROUP_N = 10

RBF_BANDWIDTH = "median"

PERM_METHODS = ["iid", "week", "biweek", "month"]

OUTPUT_REL = Path("outputs") / "drift_sanity" / "outputs"

SHAPE_COLS = [f"shape_z_h{h:02d}" for h in range(24)]
LEVEL_COLS = ["level_mean", "level_max", "level_std"]


# ============================================================
# PATHS
# ============================================================

def find_repo_root(start: Optional[Path] = None) -> Path:
    candidates: List[Path] = []
    if start is not None:
        candidates.append(start.resolve())
    try:
        candidates.append(Path(__file__).resolve())
    except NameError:
        pass
    candidates.append(Path.cwd().resolve())

    for base in candidates:
        cur = base if base.is_dir() else base.parent
        for _ in range(25):
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
# STATS HELPERS
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


# ============================================================
# BLOCK PERMUTATION
# ============================================================

def make_block_ids(dates: pd.Series, perm_method: str) -> pd.Series:
    d = pd.to_datetime(dates)
    if perm_method == "iid":
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


def block_permute_indices(block_ids: np.ndarray, nA_target: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    unique_blocks = np.unique(block_ids)
    rng.shuffle(unique_blocks)

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

    A_arr = np.array(sorted(set(A_idx)), dtype=int)
    all_idx = np.arange(len(block_ids), dtype=int)
    mask = np.ones(len(block_ids), dtype=bool)
    mask[A_arr] = False
    B_arr = all_idx[mask]
    return A_arr, B_arr


def iid_permute_indices(n: int, nA_target: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    perm = rng.permutation(np.arange(n, dtype=int))
    A = perm[:nA_target]
    B = perm[nA_target:]
    return np.array(A, dtype=int), np.array(B, dtype=int)


def run_drift_test(
    X: np.ndarray,
    idxA: np.ndarray,
    idxB: np.ndarray,
    dates: pd.Series,
    perm_method: str,
    n_perm: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    nA = len(idxA)
    nB = len(idxB)
    if nA < MIN_GROUP_N or nB < MIN_GROUP_N:
        return (np.nan, np.nan, np.nan, np.nan)

    D = _pairwise_euclidean(X)
    energy_obs = energy_stat_from_dist(D, idxA, idxB)

    # MMD bandwidth
    if RBF_BANDWIDTH == "median":
        tri = D[np.triu_indices(D.shape[0], k=1)]
        med = float(np.median(tri)) if tri.size else 1.0
        sigma = med if med > 1e-12 else 1.0
    else:
        sigma = float(RBF_BANDWIDTH)
    gamma = 1.0 / (2.0 * sigma * sigma)
    K = rbf_kernel_matrix(X, gamma)
    mmd_obs = mmd2_stat_from_kernel(K, idxA, idxB)

    e_perm = np.empty(n_perm, dtype=float)
    m_perm = np.empty(n_perm, dtype=float)
    block_ids = make_block_ids(dates, perm_method).to_numpy()

    for p in range(n_perm):
        if perm_method == "iid":
            pA, pB = iid_permute_indices(n=len(X), nA_target=nA, rng=rng)
        else:
            pA, pB = block_permute_indices(block_ids=block_ids, nA_target=nA, rng=rng)

        if len(pA) < MIN_GROUP_N or len(pB) < MIN_GROUP_N:
            e_perm[p] = np.nan
            m_perm[p] = np.nan
            continue

        e_perm[p] = energy_stat_from_dist(D, pA, pB)
        m_perm[p] = mmd2_stat_from_kernel(K, pA, pB)

    e_perm = e_perm[~np.isnan(e_perm)]
    m_perm = m_perm[~np.isnan(m_perm)]
    if e_perm.size == 0 or m_perm.size == 0:
        return (energy_obs, np.nan, mmd_obs, np.nan)

    p_energy = permutation_pvalue(energy_obs, e_perm)
    p_mmd2 = permutation_pvalue(mmd_obs, m_perm)
    return (energy_obs, p_energy, mmd_obs, p_mmd2)


# ============================================================
# DATA
# ============================================================

def load_daily_features(repo_root: Path) -> pd.DataFrame:
    p = default_daily_features_path(repo_root)
    if not p.exists():
        raise FileNotFoundError(f"Daily features parquet not found: {p}")

    df = pd.read_parquet(p).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)

    if "is_complete_24h" in df.columns:
        df = df[df["is_complete_24h"] == True].copy()  # noqa: E712

    required = ["date", "class_id"] + SHAPE_COLS + LEVEL_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Daily features missing columns: {missing}")

    return df.sort_values("date").reset_index(drop=True)


def subset_class_year(df: pd.DataFrame, class_id: str, years: Iterable[int]) -> pd.DataFrame:
    sub = df[(df["class_id"] == class_id) & (df["year"].isin(list(years)))].copy()
    return sub.sort_values("date").reset_index(drop=True)


# ============================================================
# RESULT STRUCT
# ============================================================

@dataclass(frozen=True)
class DriftRow:
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
    replicate_id: int


# ============================================================
# PLACEBO B: repeated random halves (true placebo calibration)
# ============================================================

def run_placebo_B_repeated(df: pd.DataFrame, year: int, n_repeats: int, seed: int, n_perm: int) -> List[DriftRow]:
    out: List[DriftRow] = []
    classes = sorted(df["class_id"].unique())

    for class_id in classes:
        sub = subset_class_year(df, class_id, years=[year])
        n = len(sub)
        if n < 2 * MIN_GROUP_N:
            print(f"[WARN] placebo_B_random_halves_{year} skip {class_id}: n={n}")
            continue

        dates = sub["date"]
        X_shape = sub[SHAPE_COLS].to_numpy(dtype=float)
        X_level = sub[LEVEL_COLS].to_numpy(dtype=float)

        for r in range(n_repeats):
            rng = np.random.default_rng(seed + year * 100_000 + (abs(hash(class_id)) % 10_000) * 10 + r)
            idx = np.arange(n, dtype=int)
            rng.shuffle(idx)
            nA = n // 2
            idxA = np.sort(idx[:nA])
            idxB = np.sort(idx[nA:])

            # shape
            e, p_e, m, p_m = run_drift_test(
                X=X_shape, idxA=idxA, idxB=idxB, dates=dates,
                perm_method="iid", n_perm=n_perm,
                seed=seed + 10_000 + hash((class_id, year, "shape", r)) % 10_000,
            )
            out.append(DriftRow(
                class_id=class_id,
                comparison=f"{year}_random_half_vs_half",
                placebo_type=f"placebo_B_random_halves_{year}",
                feature_space="shape_24D",
                nA=len(idxA),
                nB=len(idxB),
                stat_energy=e,
                p_energy=p_e,
                stat_mmd2=m,
                p_mmd2=p_m,
                n_perm=n_perm,
                seed=seed,
                perm_method="iid",
                replicate_id=r,
            ))

            # level
            e, p_e, m, p_m = run_drift_test(
                X=X_level, idxA=idxA, idxB=idxB, dates=dates,
                perm_method="iid", n_perm=n_perm,
                seed=seed + 20_000 + hash((class_id, year, "level", r)) % 10_000,
            )
            out.append(DriftRow(
                class_id=class_id,
                comparison=f"{year}_random_half_vs_half",
                placebo_type=f"placebo_B_random_halves_{year}",
                feature_space="level_3D",
                nA=len(idxA),
                nB=len(idxB),
                stat_energy=e,
                p_energy=p_e,
                stat_mmd2=m,
                p_mmd2=p_m,
                n_perm=n_perm,
                seed=seed,
                perm_method="iid",
                replicate_id=r,
            ))

    return out


# ============================================================
# TRAIN vs TEST block permutation (dependence-aware)
# ============================================================

def run_train_test_blockperm(df: pd.DataFrame, seed: int, n_perm: int) -> List[DriftRow]:
    out: List[DriftRow] = []
    classes = sorted(df["class_id"].unique())

    # definition of splits
    years_train = [2022, 2023]
    years_test = [2025]

    for class_id in classes:
        A = subset_class_year(df, class_id, years_train)
        B = subset_class_year(df, class_id, years_test)

        if len(A) < MIN_GROUP_N or len(B) < MIN_GROUP_N:
            print(f"[WARN] train_vs_test skip {class_id}: nA={len(A)}, nB={len(B)}")
            continue

        pooled = pd.concat([A, B], ignore_index=True)
        idxA = np.arange(len(A), dtype=int)
        idxB = np.arange(len(A), len(A) + len(B), dtype=int)
        dates = pooled["date"]

        X_shape = pooled[SHAPE_COLS].to_numpy(dtype=float)
        X_level = pooled[LEVEL_COLS].to_numpy(dtype=float)

        for perm_method in PERM_METHODS:
            # shape
            e, p_e, m, p_m = run_drift_test(
                X=X_shape, idxA=idxA, idxB=idxB, dates=dates,
                perm_method=perm_method, n_perm=n_perm,
                seed=seed + 30_000 + hash((class_id, perm_method, "shape")) % 10_000,
            )
            out.append(DriftRow(
                class_id=class_id,
                comparison="train_2022_2023_vs_test_2025",
                placebo_type="train_test_blockperm",
                feature_space="shape_24D",
                nA=len(idxA),
                nB=len(idxB),
                stat_energy=e,
                p_energy=p_e,
                stat_mmd2=m,
                p_mmd2=p_m,
                n_perm=n_perm,
                seed=seed,
                perm_method=perm_method,
                replicate_id=0,
            ))

            # level
            e, p_e, m, p_m = run_drift_test(
                X=X_level, idxA=idxA, idxB=idxB, dates=dates,
                perm_method=perm_method, n_perm=n_perm,
                seed=seed + 40_000 + hash((class_id, perm_method, "level")) % 10_000,
            )
            out.append(DriftRow(
                class_id=class_id,
                comparison="train_2022_2023_vs_test_2025",
                placebo_type="train_test_blockperm",
                feature_space="level_3D",
                nA=len(idxA),
                nB=len(idxB),
                stat_energy=e,
                p_energy=p_e,
                stat_mmd2=m,
                p_mmd2=p_m,
                n_perm=n_perm,
                seed=seed,
                perm_method=perm_method,
                replicate_id=0,
            ))

    return out


# ============================================================
# SUMMARIES (calibration + blockperm)
# ============================================================

def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (float(center - half), float(center + half))


def calibration_summary(df_placeboB: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ptype, fspace, stat), sub in df_placeboB.groupby(["placebo_type", "feature_space", "stat_name"]):
        n = len(sub)
        k005 = int((sub["pval"] < 0.05).sum())
        k001 = int((sub["pval"] < 0.01).sum())
        frac005 = k005 / n if n else np.nan
        frac001 = k001 / n if n else np.nan
        lo005, hi005 = wilson_ci(k005, n)
        lo001, hi001 = wilson_ci(k001, n)
        rows.append({
            "placebo_type": ptype,
            "feature_space": fspace,
            "stat": stat,
            "n_tests": n,
            "frac_p_lt_0p05": frac005,
            "ci95_p_lt_0p05_lo": lo005,
            "ci95_p_lt_0p05_hi": hi005,
            "frac_p_lt_0p01": frac001,
            "ci95_p_lt_0p01_lo": lo001,
            "ci95_p_lt_0p01_hi": hi001,
        })
    return pd.DataFrame(rows).sort_values(["placebo_type", "feature_space", "stat"])


def blockperm_flag_table(df_block: pd.DataFrame) -> pd.DataFrame:
    # flag: iid p<0.01 but biweek or month >0.05
    rows = []
    key_cols = ["class_id", "feature_space", "stat_name"]
    piv = df_block.pivot_table(index=key_cols, columns="perm_method", values="pval", aggfunc="first").reset_index()

    for _, r in piv.iterrows():
        p_iid = r.get("iid", np.nan)
        p_bi = r.get("biweek", np.nan)
        p_mo = r.get("month", np.nan)
        if np.isnan(p_iid):
            continue
        reasons = []
        if p_iid < 0.01:
            if not np.isnan(p_bi) and p_bi > 0.05:
                reasons.append("iid<0.01 but biweek>0.05")
            if not np.isnan(p_mo) and p_mo > 0.05:
                reasons.append("iid<0.01 but month>0.05")
        if reasons:
            rows.append({
                "class_id": r["class_id"],
                "feature_space": r["feature_space"],
                "stat": r["stat_name"],
                "p_iid": float(p_iid),
                "p_biweek": float(p_bi) if not np.isnan(p_bi) else np.nan,
                "p_month": float(p_mo) if not np.isnan(p_mo) else np.nan,
                "flag_reason": "; ".join(reasons),
            })

    return pd.DataFrame(rows).sort_values(["feature_space", "stat", "class_id"])


# ============================================================
# OUTPUT + PLOTTING
# ============================================================

def save_csv(repo_root: Path, name: str, df: pd.DataFrame) -> Path:
    out_dir = repo_root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    df.to_csv(p, index=False, encoding="utf-8")
    return p


def plot_calibration_bars(repo_root: Path, df_cal: pd.DataFrame) -> Path:
    out_dir = repo_root / OUTPUT_REL / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "placeboB_calibration_bars.png"

    # simple grouped bars: each row is (ptype, fspace, stat)
    labels = [
        f"{r.placebo_type}\n{r.feature_space}\n{r.stat}"
        for r in df_cal.itertuples(index=False)
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 0.15, df_cal["frac_p_lt_0p05"].to_numpy(), width=0.3, label="p<0.05")
    ax.bar(x + 0.15, df_cal["frac_p_lt_0p01"].to_numpy(), width=0.3, label="p<0.01")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Placebo B repeated calibration (fractions significant)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


def plot_train_test_blockperm(repo_root: Path, df_block: pd.DataFrame) -> Path:
    out_dir = repo_root / OUTPUT_REL / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "train_test_blockperm_minuslog10p.png"

    # Build a compact heatmap: rows=class (x feature/stat), cols=perm_method
    # We keep it simple: one panel per feature_space/stat
    classes = sorted(df_block["class_id"].unique())
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    panels = [
        ("shape_24D", "p_energy", axes[0, 0]),
        ("shape_24D", "p_mmd2", axes[0, 1]),
        ("level_3D", "p_energy", axes[1, 0]),
        ("level_3D", "p_mmd2", axes[1, 1]),
    ]

    for feature_space, stat_name, ax in panels:
        sub = df_block[(df_block["feature_space"] == feature_space) & (df_block["stat_name"] == stat_name)]
        mat = np.full((len(classes), len(PERM_METHODS)), np.nan, dtype=float)
        for i, c in enumerate(classes):
            for j, pm in enumerate(PERM_METHODS):
                ss = sub[(sub["class_id"] == c) & (sub["perm_method"] == pm)]
                if len(ss) > 0:
                    mat[i, j] = float(ss.iloc[0]["pval"])
        safe = np.clip(mat, 1e-12, 1.0)
        z = -np.log10(safe)
        im = ax.imshow(z, aspect="auto")
        ax.set_xticks(np.arange(len(PERM_METHODS)))
        ax.set_xticklabels(PERM_METHODS)
        ax.set_yticks(np.arange(len(classes)))
        ax.set_yticklabels(classes)
        ax.set_title(f"{feature_space} — {stat_name}  (-log10 p)")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Train (2022–2023) vs Test (2025): block permutation sensitivity", y=0.98)
    fig.tight_layout()
    fig.savefig(p, dpi=200)
    plt.close(fig)
    return p


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    repo_root = find_repo_root()
    df = load_daily_features(repo_root)

    # 1) Placebo B repeated for 2024 and 2025
    rows_b: List[DriftRow] = []
    print("Running Placebo B repeated (2024)...")
    rows_b.extend(run_placebo_B_repeated(df, year=2024, n_repeats=N_RANDOM_SPLITS_PER_YEAR, seed=SEED, n_perm=N_PERM))
    print("Running Placebo B repeated (2025)...")
    rows_b.extend(run_placebo_B_repeated(df, year=2025, n_repeats=N_RANDOM_SPLITS_PER_YEAR, seed=SEED, n_perm=N_PERM))

    df_b = pd.DataFrame([r.__dict__ for r in rows_b])
    p_b = save_csv(repo_root, "placeboB_repeated_results.csv", df_b)

    # Convert to long form for calibration summary
    long_rows = []
    for r in rows_b:
        long_rows.append({"placebo_type": r.placebo_type, "feature_space": r.feature_space, "stat_name": "p_energy", "pval": r.p_energy})
        long_rows.append({"placebo_type": r.placebo_type, "feature_space": r.feature_space, "stat_name": "p_mmd2", "pval": r.p_mmd2})
    df_b_long = pd.DataFrame(long_rows).dropna()
    df_cal = calibration_summary(df_b_long)
    p_cal = save_csv(repo_root, "placeboB_repeated_calibration_summary.csv", df_cal)

    # 2) Train vs test block permutation sensitivity
    print("Running Train vs Test block permutation (2022–2023 vs 2025)...")
    rows_tt = run_train_test_blockperm(df, seed=SEED, n_perm=N_PERM)
    df_tt = pd.DataFrame([r.__dict__ for r in rows_tt])
    p_tt = save_csv(repo_root, "train_test_blockperm_results.csv", df_tt)

    # Make long version for flags/plots
    long_tt = []
    for r in rows_tt:
        long_tt.append({"class_id": r.class_id, "feature_space": r.feature_space, "perm_method": r.perm_method, "stat_name": "p_energy", "pval": r.p_energy})
        long_tt.append({"class_id": r.class_id, "feature_space": r.feature_space, "perm_method": r.perm_method, "stat_name": "p_mmd2", "pval": r.p_mmd2})
    df_tt_long = pd.DataFrame(long_tt).dropna()
    flags = blockperm_flag_table(df_tt_long)
    p_flags = save_csv(repo_root, "train_test_blockperm_flags.csv", flags)

    # Plots
    p_plot_cal = plot_calibration_bars(repo_root, df_cal)
    p_plot_tt = plot_train_test_blockperm(repo_root, df_tt_long)

    # Console guidance (go/no-go)
    print("\n=== Outputs ===")
    print(f"[OK] {p_b}")
    print(f"[OK] {p_cal}")
    print(f"[OK] {p_tt}")
    print(f"[OK] {p_flags}")
    print(f"[OK] {p_plot_cal}")
    print(f"[OK] {p_plot_tt}")

    print("\n=== How to interpret (go/no-go for 5b) ===")
    print("Placebo B repeated should have roughly ~5% below 0.05 and ~1% below 0.01.")
    print("If frac_p_lt_0p01 is consistently > 0.05 or frac_p_lt_0p05 > 0.12, investigate artifacts/dependence.")
    print("Train-vs-test blockperm: if drift stays strong under week/biweek for most classes, inference is robust.")
    print("If significance disappears already at biweek everywhere, treat iid p-values as too optimistic and rely on block-perm in write-up.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
