from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

SEED = 12345
LOCAL_TRAIN_END = pd.Timestamp("2023-12-31")

# Split definition (consistent with your design)
TRAIN_YEARS = (2022, 2023)
VALIDATE_YEARS = (2024,)
TEST_YEARS = (2025,)

# Distance metric: correlation distance on z-normalised 24-hour shapes
SHAPE_COLS = [f"shape_z_h{h:02d}" for h in range(24)]

# Strategy B weighting
BETA = 0.5  # mixture of uniform + recency
MONTHS_PER_DAY = 1.0 / 30.4375  # approx average month length

# Pruning rule (for B)
# Interpret as "minimum effective support mass equal to X full-weight days"
MIN_SUPPORT_EQ_DAYS = 3.0

# Safety: keep at least this many medoids per class even if thresholds prune more
MIN_KEEP_PER_CLASS = 2

# Outputs
OUTPUT_REL = (
    Path("1_Configs")
    / "1.2_Data_processed"
    / "DA-market"
    / "99_reports"
    / "da_comparison_A_vs_B"
)


# ============================================================
# PATH HELPERS
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
        for _ in range(30):
            if (cur / "1_Configs").exists():
                return cur
            if cur.parent == cur:
                break
            cur = cur.parent
    raise FileNotFoundError("Could not locate repo root (no '1_Configs/' found).")


def find_file(repo_root: Path, filename: str) -> Path:
    """
    Search for filename in common processed output folders.
    If you store regimes somewhere else, adjust SEARCH_DIRS.
    """
    search_dirs = [
        repo_root / "1_Configs" / "1.2_Data_processed" / "DA-market",
        repo_root / "outputs",
        repo_root,
    ]
    hits: List[Path] = []
    for d in search_dirs:
        if d.exists():
            hits.extend(list(d.rglob(filename)))
    if not hits:
        raise FileNotFoundError(
            f"Could not find '{filename}' under:\n" + "\n".join(str(d) for d in search_dirs)
        )
    # if multiple hits, take the most recently modified
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def default_daily_features_path(repo_root: Path) -> Path:
    return (
        repo_root
        / "1_Configs"
        / "1.2_Data_processed"
        / "DA-market"
        / "01_daily_features"
        / "da_daily_features_scoped_2022_2025.parquet"
    )


def ensure_out_dir(repo_root: Path) -> Path:
    out_dir = repo_root / OUTPUT_REL
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(parents=True, exist_ok=True)
    return out_dir


# ============================================================
# CORE MATH
# ============================================================

def corr_distance_to_set(x: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Correlation distance: 1 - corr(x, m)
    Assumes x and rows of M are 1D vectors (length 24).
    If vectors are z-normalised per day, correlation is meaningful.
    """
    # Robust correlation computed via dot-product normalised by norms
    x0 = x - x.mean()
    xn = np.linalg.norm(x0)
    if xn < 1e-12:
        return np.ones(M.shape[0], dtype=float)

    M0 = M - M.mean(axis=1, keepdims=True)
    Mn = np.linalg.norm(M0, axis=1)
    # Avoid divide-by-zero
    Mn = np.where(Mn < 1e-12, np.nan, Mn)

    corr = (M0 @ x0) / (Mn * xn)
    corr = np.nan_to_num(corr, nan=0.0)
    return 1.0 - corr


def recency_weight(day: pd.Timestamp, anchor: pd.Timestamp, half_life_months: float, beta: float) -> float:
    """
    w = beta * 1 + (1-beta) * exp(-ln(2) * age / half_life)
    where age is in months from day to anchor (non-negative for days <= anchor).
    """
    age_days = (anchor - day).days
    age_months = max(0.0, float(age_days) * MONTHS_PER_DAY)
    if half_life_months <= 0:
        decay = 0.0
    else:
        decay = float(np.exp(-np.log(2.0) * age_months / half_life_months))
    return float(beta + (1.0 - beta) * decay)


# ============================================================
# DATA LOADING
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

    missing = [c for c in ["date", "class_id"] + SHAPE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Daily features missing required columns: {missing}")

    df["split"] = "other"
    df.loc[df["year"].isin(TRAIN_YEARS), "split"] = "train"
    df.loc[df["year"].isin(VALIDATE_YEARS), "split"] = "validate"
    df.loc[df["year"].isin(TEST_YEARS), "split"] = "test"

    return df.sort_values(["class_id", "date"]).reset_index(drop=True)


def load_half_life_by_class(repo_root: Path) -> Dict[str, float]:
    p = find_file(repo_root, "B_best_half_life_by_class.json")
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    # ensure float
    return {k: float(v) for k, v in obj.items()}

def load_strategy_A_library(repo_root: Path) -> pd.DataFrame:
    # probabilities file (pruned)
    p_prob = find_file(repo_root, "A_medoids_probabilities_pruned.csv")
    df_prob = pd.read_csv(p_prob)

    if "medoid_id" not in df_prob.columns or "class_id" not in df_prob.columns:
        raise ValueError(
            "A_medoids_probabilities_pruned.csv must contain at least ['class_id','medoid_id'] "
            f"but has: {list(df_prob.columns)}"
        )

    # shapes file containing dates
    p_shapes = find_file(repo_root, "A_medoids_shapes.csv")
    df_shapes = pd.read_csv(p_shapes)

    # Required mapping columns
    if "medoid_id" not in df_shapes.columns:
        raise ValueError(
            "A_medoids_shapes.csv must contain 'medoid_id' to map to dates. "
            f"Found: {list(df_shapes.columns)}"
        )

    # Detect date column name
    date_col = None
    for c in ["medoid_date", "date", "day", "medoid_day"]:
        if c in df_shapes.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(
            "A_medoids_shapes.csv must contain a medoid date column (medoid_date/date/day/medoid_day). "
            f"Found: {list(df_shapes.columns)}"
        )

    df_shapes = df_shapes.rename(columns={date_col: "medoid_date"}).copy()
    df_shapes["medoid_date"] = pd.to_datetime(df_shapes["medoid_date"])

    # If shapes file has class_id, great; if not, we rely on probabilities file class_id
    keep_cols = ["medoid_id", "medoid_date"]
    if "class_id" in df_shapes.columns:
        keep_cols.append("class_id")

    df_shapes = df_shapes[keep_cols].drop_duplicates()

    # Merge: attach medoid_date to each (class_id, medoid_id)
    dfA = df_prob.merge(df_shapes, on="medoid_id", how="left", suffixes=("", "_shapes"))

    # If shapes provided class_id, ensure it matches when present
    if "class_id_shapes" in dfA.columns:
        mismatch = dfA["class_id_shapes"].notna() & (dfA["class_id_shapes"] != dfA["class_id"])
        if mismatch.any():
            bad = dfA.loc[mismatch, ["class_id", "class_id_shapes", "medoid_id"]].head(10)
            raise ValueError(f"medoid_id→class_id mismatch between A files. Examples:\n{bad}")
        dfA = dfA.drop(columns=["class_id_shapes"])

    if dfA["medoid_date"].isna().any():
        missing = dfA[dfA["medoid_date"].isna()][["class_id", "medoid_id"]].drop_duplicates().head(10)
        raise ValueError(
            "Some A medoids could not be mapped to dates via A_medoids_shapes.csv. "
            f"Examples:\n{missing}"
        )

    # Use medoid_id as the regime identifier (cluster_id not required)
    dfA = dfA.rename(columns={"medoid_id": "regime_id"}).copy()
    return dfA



def load_strategy_B_library(repo_root: Path) -> pd.DataFrame:
    # We accept either already-pruned or raw; 5b will output pruned anyway.
    # Prefer the non-pruned B library if it exists.
    try:
        p = find_file(repo_root, "B_medoids_probabilities.csv")
    except FileNotFoundError:
        p = find_file(repo_root, "B_medoids_probabilities_raw.csv")
    df = pd.read_csv(p)
    required = {"class_id", "cluster_id"}
    if not required.issubset(df.columns):
        raise ValueError(f"B library missing columns {required}. Found: {list(df.columns)}")
    date_col = None
    for c in ["medoid_date", "date", "medoid_day"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError("B library must contain a medoid date column (medoid_date/date/medoid_day).")
    df = df.rename(columns={date_col: "medoid_date"}).copy()
    df["medoid_date"] = pd.to_datetime(df["medoid_date"])
    return df


# ============================================================
# PRUNING B: compute support from train assignments
# ============================================================

@dataclass(frozen=True)
class PruneDecision:
    class_id: str
    cluster_id: int
    medoid_date: pd.Timestamp
    support_mass: float
    n_assigned: int
    keep: bool
    reason: str


def compute_medoid_shapes(
    daily: pd.DataFrame,
    lib: pd.DataFrame,
) -> Dict[str, Tuple[np.ndarray, List[int], List[pd.Timestamp]]]:
    """
    Returns dict:
      class_id -> (M, cluster_ids, medoid_dates)
    where M is array shape (K, 24)
    """
    out: Dict[str, Tuple[np.ndarray, List[int], List[pd.Timestamp]]] = {}
    for class_id, sub in lib.groupby("class_id"):
        rows = []
        cids: List[int] = []
        mdates: List[pd.Timestamp] = []
        for r in sub.itertuples(index=False):
            mdate = pd.to_datetime(getattr(r, "medoid_date"))
            cid = int(getattr(r, "cluster_id"))
            day = daily[(daily["class_id"] == class_id) & (daily["date"] == mdate)]
            if day.empty:
                raise ValueError(f"Medoid date not found in daily features: class={class_id}, date={mdate.date()}")
            rows.append(day.iloc[0][SHAPE_COLS].to_numpy(dtype=float))
            cids.append(cid)
            mdates.append(mdate)
        M = np.vstack(rows) if rows else np.empty((0, 24), dtype=float)
        out[class_id] = (M, cids, mdates)
    return out


def prune_B_by_support(
    daily: pd.DataFrame,
    libB: pd.DataFrame,
    half_life_by_class: Dict[str, float],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    - Assign each TRAIN day to nearest B medoid within its class (corr distance).
    - Compute recency weight for each day.
    - Support mass per medoid = sum weights of assigned days.
    - Prune medoids with support_mass < MIN_SUPPORT_EQ_DAYS (approx; max weight ~1).
    - Keep at least MIN_KEEP_PER_CLASS (highest support) regardless.
    Returns:
      pruned_library_df, prune_report_df
    """
    med_shapes = compute_medoid_shapes(daily, libB)

    decisions: List[PruneDecision] = []
    pruned_rows: List[Dict[str, object]] = []

    train = daily[daily["split"] == "train"].copy()

    for class_id in sorted(libB["class_id"].unique()):
        if class_id not in med_shapes:
            continue

        M, cluster_ids, medoid_dates = med_shapes[class_id]
        if M.shape[0] == 0:
            continue

        train_c = train[train["class_id"] == class_id].copy()
        if train_c.empty:
            print(f"[WARN] No train days for class {class_id}; skipping pruning.")
            continue

        hl = float(half_life_by_class.get(class_id, 24.0))
        # compute weights
        w = np.array([recency_weight(d, LOCAL_TRAIN_END, hl, BETA) for d in train_c["date"]], dtype=float)
        X = train_c[SHAPE_COLS].to_numpy(dtype=float)

        # Assign each day to nearest medoid
        assigned_idx = np.empty(len(train_c), dtype=int)
        min_dist = np.empty(len(train_c), dtype=float)

        for i in range(len(train_c)):
            dists = corr_distance_to_set(X[i], M)
            j = int(np.argmin(dists))
            assigned_idx[i] = j
            min_dist[i] = float(dists[j])

        # Support per medoid
        K = M.shape[0]
        support = np.zeros(K, dtype=float)
        counts = np.zeros(K, dtype=int)
        for j in range(K):
            mask = assigned_idx == j
            support[j] = float(w[mask].sum())
            counts[j] = int(mask.sum())

        # Determine keep/prune
        # Primary: minimum support mass threshold
        keep = support >= float(MIN_SUPPORT_EQ_DAYS)

        # Safety: keep at least MIN_KEEP_PER_CLASS
        if keep.sum() < MIN_KEEP_PER_CLASS:
            order = np.argsort(-support)  # descending support
            keep[:] = False
            keep[order[:MIN_KEEP_PER_CLASS]] = True

        # Build decision rows
        for j in range(K):
            reason = "support_ok"
            if not keep[j]:
                reason = f"support_mass<{MIN_SUPPORT_EQ_DAYS:g}"
            decisions.append(PruneDecision(
                class_id=class_id,
                cluster_id=int(cluster_ids[j]),
                medoid_date=pd.to_datetime(medoid_dates[j]),
                support_mass=float(support[j]),
                n_assigned=int(counts[j]),
                keep=bool(keep[j]),
                reason=reason,
            ))

        # Build pruned library for this class
        kept_idx = np.where(keep)[0]
        kept_support = support[kept_idx]
        total = float(kept_support.sum())
        if total <= 0:
            # Fallback: keep the single best
            best = int(np.argmax(support))
            kept_idx = np.array([best], dtype=int)
            kept_support = support[kept_idx]
            total = float(kept_support.sum())

        probs = kept_support / total

        for rank, j in enumerate(kept_idx):
            pruned_rows.append({
                "strategy": "B",
                "class_id": class_id,
                "cluster_id": int(cluster_ids[j]),
                "medoid_date": pd.to_datetime(medoid_dates[j]).date().isoformat(),
                "probability": float(probs[rank]),
                "support_mass_train": float(support[j]),
                "n_assigned_train_days": int(counts[j]),
                "half_life_months": float(hl),
                "beta": float(BETA),
                "k_kept_in_class": int(len(kept_idx)),
            })

    df_pruned = pd.DataFrame(pruned_rows).sort_values(["class_id", "cluster_id"]).reset_index(drop=True)
    df_report = pd.DataFrame([d.__dict__ for d in decisions]).sort_values(["class_id", "keep", "support_mass"], ascending=[True, True, False])

    # Save
    df_pruned.to_csv(out_dir / "B_medoids_probabilities_pruned.csv", index=False, encoding="utf-8")
    df_report.to_csv(out_dir / "B_pruning_report.csv", index=False, encoding="utf-8")

    # Simple summary
    summ = (
        df_report.groupby(["class_id", "keep"])["cluster_id"]
        .count()
        .reset_index()
        .pivot(index="class_id", columns="keep", values="cluster_id")
        .fillna(0)
        .rename(columns={False: "n_pruned", True: "n_kept"})
        .reset_index()
    )
    summ.to_csv(out_dir / "B_pruning_summary_by_class.csv", index=False, encoding="utf-8")

    return df_pruned, df_report


# ============================================================
# A vs B COMPARISON: representation error
# ============================================================

def load_library_shapes_from_pruned(
    daily: pd.DataFrame,
    lib_pruned: pd.DataFrame,
) -> Dict[str, np.ndarray]:
    """
    lib_pruned columns must include: class_id, medoid_date.
    Returns class_id -> M shape (K, 24)
    """
    out: Dict[str, np.ndarray] = {}
    for class_id, sub in lib_pruned.groupby("class_id"):
        rows = []
        for r in sub.itertuples(index=False):
            mdate = pd.to_datetime(getattr(r, "medoid_date"))
            day = daily[(daily["class_id"] == class_id) & (daily["date"] == mdate)]
            if day.empty:
                raise ValueError(f"Medoid date not found in daily features: class={class_id}, date={mdate.date()}")
            rows.append(day.iloc[0][SHAPE_COLS].to_numpy(dtype=float))
        out[class_id] = np.vstack(rows) if rows else np.empty((0, 24), dtype=float)
    return out


def representation_error(
    daily: pd.DataFrame,
    shapes_by_class: Dict[str, np.ndarray],
    strategy_name: str,
    split: str,
) -> pd.DataFrame:
    """
    For each day in split, compute min correlation distance to medoids of its class.
    Return per-class summary stats.
    """
    sub = daily[daily["split"] == split].copy()
    rows: List[Dict[str, object]] = []
    for class_id, dsub in sub.groupby("class_id"):
        M = shapes_by_class.get(class_id, np.empty((0, 24), dtype=float))
        if M.shape[0] == 0:
            continue

        X = dsub[SHAPE_COLS].to_numpy(dtype=float)
        mins = np.empty(len(dsub), dtype=float)
        for i in range(len(dsub)):
            dists = corr_distance_to_set(X[i], M)
            mins[i] = float(np.min(dists))

        rows.append({
            "strategy": strategy_name,
            "split": split,
            "class_id": class_id,
            "n_days": int(len(dsub)),
            "mean_min_dist": float(np.mean(mins)),
            "p90_min_dist": float(np.quantile(mins, 0.90)),
            "p95_min_dist": float(np.quantile(mins, 0.95)),
            "p99_min_dist": float(np.quantile(mins, 0.99)),
            "worst_1pct_mean": float(np.mean(np.sort(mins)[max(0, int(0.99 * len(mins))):])),
        })
    return pd.DataFrame(rows).sort_values(["split", "class_id", "strategy"]).reset_index(drop=True)


def add_overall_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for (strategy, split), sub in out.groupby(["strategy", "split"]):
        n = sub["n_days"].sum()
        if n <= 0:
            continue
        w = sub["n_days"].to_numpy(dtype=float)
        w = w / w.sum()
        out = pd.concat([out, pd.DataFrame([{
            "strategy": strategy,
            "split": split,
            "class_id": "ALL",
            "n_days": int(n),
            "mean_min_dist": float(np.sum(w * sub["mean_min_dist"])),
            "p90_min_dist": float(np.sum(w * sub["p90_min_dist"])),
            "p95_min_dist": float(np.sum(w * sub["p95_min_dist"])),
            "p99_min_dist": float(np.sum(w * sub["p99_min_dist"])),
            "worst_1pct_mean": float(np.sum(w * sub["worst_1pct_mean"])),
        }])], ignore_index=True)
    return out.reset_index(drop=True)


def plot_compare(repo_root: Path, out_dir: Path, df_cmp: pd.DataFrame) -> None:
    # Bar plot: mean_min_dist per class for validate and test
    for split in ["validate", "test"]:
        sub = df_cmp[(df_cmp["split"] == split) & (df_cmp["class_id"] != "ALL")].copy()
        if sub.empty:
            continue

        classes = sorted(sub["class_id"].unique())
        strategies = ["A", "B"]

        x = np.arange(len(classes))
        width = 0.35

        fig, ax = plt.subplots(figsize=(10, 4))
        for i, strat in enumerate(strategies):
            ssub = sub[sub["strategy"] == strat].set_index("class_id").reindex(classes)
            ax.bar(x + (i - 0.5) * width, ssub["mean_min_dist"].to_numpy(), width=width, label=strat)

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_title(f"Representation error (mean min corr-distance) — {split}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "plots" / f"AB_mean_min_dist_{split}.png", dpi=200)
        plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    repo_root = find_repo_root()
    out_dir = ensure_out_dir(repo_root)

    print("[INFO] Loading daily features...")
    daily = load_daily_features(repo_root)

    print("[INFO] Loading Strategy A library (pruned)...")
    libA = load_strategy_A_library(repo_root)
    libA_pruned = libA[["class_id", "regime_id", "medoid_date"]].copy()
    libA_pruned["medoid_date"] = pd.to_datetime(libA_pruned["medoid_date"])

    print("[INFO] Loading Strategy B library (raw)...")
    libB = load_strategy_B_library(repo_root)

    print("[INFO] Loading best half-life per class...")
    hl_by_class = load_half_life_by_class(repo_root)

    # ---- 5b: prune B
    print("[INFO] Pruning Strategy B by effective support mass...")
    dfB_pruned, dfB_report = prune_B_by_support(daily, libB, hl_by_class, out_dir)
    print("[OK] Wrote:", out_dir / "B_medoids_probabilities_pruned.csv")

    # ---- Compare A vs B on validate/test
    print("[INFO] Comparing representation errors A vs B on validate/test...")
    # Build shapes for A
    shapesA = load_library_shapes_from_pruned(daily, libA_pruned[["class_id", "medoid_date"]])
    # Build shapes for B (pruned)
    dfB_for_shapes = dfB_pruned[["class_id", "medoid_date"]].copy()
    dfB_for_shapes["medoid_date"] = pd.to_datetime(dfB_for_shapes["medoid_date"])
    shapesB = load_library_shapes_from_pruned(daily, dfB_for_shapes)

    dfA_val = representation_error(daily, shapesA, "A", "validate")
    dfA_tst = representation_error(daily, shapesA, "A", "test")
    dfB_val = representation_error(daily, shapesB, "B", "validate")
    dfB_tst = representation_error(daily, shapesB, "B", "test")

    df_cmp = pd.concat([dfA_val, dfA_tst, dfB_val, dfB_tst], ignore_index=True)
    df_cmp = add_overall_row(df_cmp)
    df_cmp.to_csv(out_dir / "AB_representation_error_summary.csv", index=False, encoding="utf-8")
    print("[OK] Wrote:", out_dir / "AB_representation_error_summary.csv")

    # Extra: counts of regimes per class
    regA = libA_pruned.groupby("class_id")["cluster_id"].nunique().reset_index(name="A_n_regimes")
    regB = dfB_pruned.groupby("class_id")["cluster_id"].nunique().reset_index(name="B_n_regimes")
    reg = pd.merge(regA, regB, on="class_id", how="outer").fillna(0)
    reg.to_csv(out_dir / "AB_regime_counts_by_class.csv", index=False, encoding="utf-8")

    # Plots
    plot_compare(repo_root, out_dir, df_cmp)

    # Console summary (quick read)
    all_rows = df_cmp[df_cmp["class_id"] == "ALL"].sort_values(["split", "strategy"])
    print("\n=== OVERALL (ALL classes; weighted by n_days) ===")
    for r in all_rows.itertuples(index=False):
        print(
            f"{r.split:8s}  {r.strategy}  mean={r.mean_min_dist:.4f}  "
            f"p95={r.p95_min_dist:.4f}  worst1%={r.worst_1pct_mean:.4f}"
        )

    print("\n=== Notes ===")
    print("- Lower distances are better (days are closer to some medoid).")
    print("- Compare A vs B on TEST (2025) especially: B should improve if drift exists.")
    print("- If B improves means but worsens tail (worst1%), pruning threshold may be too aggressive or too lenient.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
