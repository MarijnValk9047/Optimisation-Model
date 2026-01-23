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
N_HOURS = 24

# If True, also compute K-common comparison (fairness when kept_k differs after pruning)
DO_K_COMMON = True

# If True, keep a day-level parquet with assignments and distances (can be large but useful)
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
# DATA INFERENCE
# =========================

def infer_id_col(df: pd.DataFrame) -> str:
    for c in ["medoid_id", "cluster_id", "prototype_id"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not infer id column in df. Columns: {list(df.columns)}")


def infer_prob_col(df: pd.DataFrame) -> str:
    for c in ["p_medoid_given_class_pruned", "p_medoid_given_class", "prob", "p", "probability"]:
        if c in df.columns:
            return c
    raise KeyError(f"Could not infer probability column in df. Columns: {list(df.columns)}")


def standardize_rows(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    return (X - mu) / sd


def corr_distance_matrix(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Pearson correlation distance 1 - corr, computed efficiently by standardising rows.
    """
    Xz = standardize_rows(X.astype(float))
    Mz = standardize_rows(M.astype(float))

    # corr with ddof=0 => mean(xz* mz) over 24 points
    corr = (Xz @ Mz.T) / float(N_HOURS)
    corr = np.clip(corr, -1.0, 1.0)
    return 1.0 - corr


@dataclass(frozen=True)
class Library:
    name: str
    medoids: pd.DataFrame
    probs: pd.DataFrame
    id_col: str
    prob_col: str


def load_library(repo_root: Path, strategy: str) -> Library:
    """
    strategy: "A" or "B"
    Expected files:
      - strategy_A/A_medoids_shapes.parquet
      - strategy_A/A_medoids_probabilities_pruned.(parquet|csv)
      - strategy_B/B_medoids_shapes.parquet
      - strategy_B/B_medoids_probabilities_pruned.(parquet|csv)
    """
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

    if "class_id" not in medoids.columns:
        raise KeyError(f"{strategy} medoids missing 'class_id'. Columns: {list(medoids.columns)}")
    for c in SHAPE_COLS:
        if c not in medoids.columns:
            raise KeyError(f"{strategy} medoids missing shape col {c}")

    if "class_id" not in probs.columns:
        raise KeyError(f"{strategy} probs missing 'class_id'. Columns: {list(probs.columns)}")

    id_col = infer_id_col(probs)
    prob_col = infer_prob_col(probs)

    return Library(name=name, medoids=medoids, probs=probs, id_col=id_col, prob_col=prob_col)


# =========================
# EVALUATION
# =========================

def select_top_k_by_prob(probs: pd.DataFrame, id_col: str, prob_col: str, k: int) -> pd.DataFrame:
    if k <= 0:
        return probs.iloc[0:0].copy()
    return (
        probs.sort_values(prob_col, ascending=False)
        .head(k)
        .copy()
    )


def eval_library_on_test(
    test_df: pd.DataFrame,
    lib: Library,
    k_common_by_class: Dict[str, int] | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - summary_by_class (one row per class)
      - day_level (one row per day) with nearest-medoid and expected distance
    """
    test_df = test_df.copy()
    test_df["date"] = pd.to_datetime(test_df["date"]).dt.date

    summaries: List[Dict] = []
    day_rows: List[Dict] = []

    classes = sorted(test_df["class_id"].unique())
    for class_id in classes:
        days = test_df[test_df["class_id"] == class_id].copy()
        if len(days) == 0:
            continue

        # probs for this class
        probs_c = lib.probs[lib.probs["class_id"] == class_id].copy()
        if len(probs_c) == 0:
            summaries.append({
                "strategy": lib.name,
                "class_id": class_id,
                "n_test_days": len(days),
                "kept_k": 0,
                "note": "no_probs_for_class",
            })
            continue

        # If K-common requested, restrict to top-k for this class
        if k_common_by_class is not None:
            k_use = int(k_common_by_class.get(class_id, 0))
            probs_c = select_top_k_by_prob(probs_c, lib.id_col, lib.prob_col, k=k_use)

        kept_k = int(len(probs_c))

        # medoids subset aligned to probs ids
        ids = probs_c[lib.id_col].astype(int).tolist()
        med_c = lib.medoids[(lib.medoids["class_id"] == class_id)].copy()

        # medoids may contain more than pruned probs; keep only those in probs
        # try to align medoid identifier
        med_id_col = "medoid_id" if "medoid_id" in med_c.columns else ("cluster_id" if "cluster_id" in med_c.columns else None)
        if med_id_col is None:
            raise KeyError(f"{lib.name} medoids missing medoid_id/cluster_id columns: {list(med_c.columns)}")

        med_c = med_c[med_c[med_id_col].astype(int).isin(ids)].copy()
        if len(med_c) == 0:
            summaries.append({
                "strategy": lib.name,
                "class_id": class_id,
                "n_test_days": len(days),
                "kept_k": kept_k,
                "note": "no_medoids_for_probs_ids",
            })
            continue

        # order medoids in the same order as probs ids for expected distance
        med_c[med_id_col] = med_c[med_id_col].astype(int)
        med_c = med_c.set_index(med_id_col).loc[ids].reset_index()

        X = days[SHAPE_COLS].to_numpy(dtype=float)
        M = med_c[SHAPE_COLS].to_numpy(dtype=float)

        D = corr_distance_matrix(X, M)  # shape: (n_days, k)

        # nearest
        argmin = np.argmin(D, axis=1)
        d_min = D[np.arange(len(days)), argmin]
        nearest_id = np.array(ids, dtype=int)[argmin]

        # expected distance under probs
        p_vec = probs_c[lib.prob_col].to_numpy(dtype=float)
        p_sum = float(p_vec.sum())
        if p_sum <= 0.0:
            p_vec = np.ones_like(p_vec) / float(len(p_vec))
        else:
            p_vec = p_vec / p_sum
        d_exp = D @ p_vec

        # summary stats
        def q(arr: np.ndarray, qq: float) -> float:
            return float(np.quantile(arr, qq))

        summaries.append({
            "strategy": lib.name,
            "class_id": class_id,
            "n_test_days": int(len(days)),
            "kept_k": int(len(ids)),
            "mean_dmin": float(d_min.mean()),
            "median_dmin": float(np.median(d_min)),
            "p90_dmin": q(d_min, 0.90),
            "p95_dmin": q(d_min, 0.95),
            "max_dmin": float(d_min.max()),
            "mean_dexp": float(d_exp.mean()),
            "p95_dexp": q(d_exp, 0.95),
            "note": "",
        })

        for i, row in enumerate(days.itertuples(index=False)):
            day_rows.append({
                "strategy": lib.name,
                "date": row.date,
                "class_id": class_id,
                "nearest_medoid_id": int(nearest_id[i]),
                "d_min": float(d_min[i]),
                "d_exp": float(d_exp[i]),
            })

    return pd.DataFrame(summaries), pd.DataFrame(day_rows)


def compare_A_vs_B(summary_A: pd.DataFrame, summary_B: pd.DataFrame) -> pd.DataFrame:
    merged = summary_A.merge(summary_B, on="class_id", how="outer", suffixes=("_A", "_B"))
    # win flags (lower is better)
    for metric in ["mean_dmin", "p95_dmin", "mean_dexp", "p95_dexp"]:
        a = merged.get(metric + "_A")
        b = merged.get(metric + "_B")
        if a is not None and b is not None:
            merged["B_better_" + metric] = (b < a)
    return merged


def plot_bar(repo_root: Path, df_cmp: pd.DataFrame, title: str, filename: str, metric: str) -> None:
    out = reports_dir(repo_root) / filename
    classes = df_cmp["class_id"].astype(str).tolist()
    a = df_cmp[f"{metric}_A"].to_numpy(dtype=float)
    b = df_cmp[f"{metric}_B"].to_numpy(dtype=float)

    x = np.arange(len(classes))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x - 0.2, a, width=0.4, label="A_pruned")
    ax.bar(x + 0.2, b, width=0.4, label="B_pruned")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=20, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main() -> int:
    repo_root = find_repo_root(Path(__file__).parent)

    df = pd.read_parquet(daily_features_path(repo_root)).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)

    # Use complete days only if column exists
    if "is_complete_24h" in df.columns:
        df = df[df["is_complete_24h"] == True].copy()  # noqa: E712

    test = df[df["year"] == TEST_YEAR].copy()
    if len(test) == 0:
        raise ValueError(f"No test rows found for year {TEST_YEAR}")

    libA = load_library(repo_root, "A")
    libB = load_library(repo_root, "B")

    # ---------
    # AS-IS comparison
    # ---------
    sumA, dayA = eval_library_on_test(test, libA, k_common_by_class=None)
    sumB, dayB = eval_library_on_test(test, libB, k_common_by_class=None)

    cmp_as_is = compare_A_vs_B(sumA, sumB)
    out_cmp = reports_dir(repo_root) / "A_vs_B_pruned_test2025_comparison_as_is.csv"
    cmp_as_is.to_csv(out_cmp, index=False, encoding="utf-8")

    if WRITE_DAY_LEVEL:
        day = pd.concat([dayA, dayB], ignore_index=True)
        out_day = reports_dir(repo_root) / "A_vs_B_pruned_test2025_day_level.parquet"
        day.to_parquet(out_day, index=False)

    plot_bar(
        repo_root,
        cmp_as_is,
        title="A vs B (pruned) on 2025 — mean nearest-medoid correlation distance (as-is)",
        filename="A_vs_B_test2025_mean_dmin_as_is.png",
        metric="mean_dmin",
    )
    plot_bar(
        repo_root,
        cmp_as_is,
        title="A vs B (pruned) on 2025 — p95 nearest-medoid correlation distance (as-is)",
        filename="A_vs_B_test2025_p95_dmin_as_is.png",
        metric="p95_dmin",
    )

    # ---------
    # K-COMMON (fairness) comparison
    # ---------
    if DO_K_COMMON:
        # compute per class common K based on kept_k
        keptA = sumA[["class_id", "kept_k"]].set_index("class_id")["kept_k"].to_dict()
        keptB = sumB[["class_id", "kept_k"]].set_index("class_id")["kept_k"].to_dict()
        classes = sorted(set(keptA.keys()) | set(keptB.keys()))
        k_common = {c: int(min(keptA.get(c, 0), keptB.get(c, 0))) for c in classes}

        sumA_k, _ = eval_library_on_test(test, libA, k_common_by_class=k_common)
        sumB_k, _ = eval_library_on_test(test, libB, k_common_by_class=k_common)

        cmp_k = compare_A_vs_B(sumA_k, sumB_k)
        out_cmp_k = reports_dir(repo_root) / "A_vs_B_pruned_test2025_comparison_k_common.csv"
        cmp_k.to_csv(out_cmp_k, index=False, encoding="utf-8")

        plot_bar(
            repo_root,
            cmp_k,
            title="A vs B (pruned) on 2025 — mean nearest-medoid corr distance (K-common fairness)",
            filename="A_vs_B_test2025_mean_dmin_k_common.png",
            metric="mean_dmin",
        )
        plot_bar(
            repo_root,
            cmp_k,
            title="A vs B (pruned) on 2025 — p95 nearest-medoid corr distance (K-common fairness)",
            filename="A_vs_B_test2025_p95_dmin_k_common.png",
            metric="p95_dmin",
        )

    print("=== Done ===")
    print(f"Wrote: {out_cmp}")
    if WRITE_DAY_LEVEL:
        print(f"Wrote: {out_day}")
    if DO_K_COMMON:
        print(f"Wrote: {out_cmp_k}")
    print(f"Wrote plots to: {reports_dir(repo_root)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
