from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# ---------------------------
# USER-EDITABLE CONFIG
# ---------------------------

EXPECTED_FILES_STEMS = [
    "GUI_ENERGY_PRICES_202112312300-202212312300",
    "GUI_ENERGY_PRICES_202212312300-202312312300",
    "GUI_ENERGY_PRICES_202312312300-202412312300",
    "GUI_ENERGY_PRICES_202412312300-202512312300",
]

RAW_DIR_CANDIDATES = [
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\1_Configs\1.1_Data_raw\DA-market",
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\Configs\Data_raw\DA-market",
]

PROCESSED_DIR = r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\1_Configs\1.2_Data_processed\DA-market\00_harmonised"
LOGS_ROOT = r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\4_logs\da_regimes"

REQUIRED_COLUMNS = ["MTU (CET/CEST)", "Day-ahead Price (EUR/MWh)"]

LOCAL_TZ = "Europe/Amsterdam"  # matches CET/CEST label


# ---------------------------
# INTERNAL HELPERS
# ---------------------------

def find_existing_dir(candidates: List[str]) -> Path:
    for c in candidates:
        p = Path(c)
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError("No RAW_DIR_CANDIDATES exist; update config at top of file.")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_expected_files(raw_dir: Path) -> List[Path]:
    found: List[Path] = []
    for stem in EXPECTED_FILES_STEMS:
        exact = raw_dir / stem
        csv_path = raw_dir / f"{stem}.csv"
        if exact.exists():
            found.append(exact)
        elif csv_path.exists():
            found.append(csv_path)
        else:
            raise FileNotFoundError(f"Missing raw file: {stem} (or {stem}.csv) in {raw_dir}")
    return found


def sniff_dialect(sample_text: str) -> Tuple[str, str]:
    """
    Returns (delimiter, decimal_symbol_guess).
    We guess decimal based on presence of ',' in numeric-looking tokens.
    """
    try:
        dialect = csv.Sniffer().sniff(sample_text, delimiters=[",", ";", "\t", "|"])
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ";"  # common NL export

    # decimal guess: if delimiter is ';', decimals are often ','.
    # But price can also be dot. We'll handle both robustly later.
    decimal_guess = "," if delimiter == ";" else "."
    return delimiter, decimal_guess


def read_price_file(path: Path) -> pd.DataFrame:
    # Read small sample for delimiter sniffing
    with path.open("r", encoding="utf-8", errors="replace") as f:
        sample = f.read(10_000)

    delimiter, _ = sniff_dialect(sample)

    # First try: default decimal (pandas can handle decimal=',' if needed)
    # We will coerce price afterwards anyway (replace comma -> dot).
    df = pd.read_csv(path, sep=delimiter, engine="python")

    # Basic schema check
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name}: missing columns {missing_cols}. Found columns: {list(df.columns)[:20]}")

    df = df[REQUIRED_COLUMNS].copy()
    df.rename(
        columns={
            "MTU (CET/CEST)": "mtu",
            "Day-ahead Price (EUR/MWh)": "price_eur_mwh",
        },
        inplace=True,
    )

    # Price cleaning: handle commas, stray spaces, non-numeric
    df["price_eur_mwh"] = (
        df["price_eur_mwh"]
        .astype(str)
        .str.strip()
        .str.replace(",", ".", regex=False)
    )
    df["price_eur_mwh"] = pd.to_numeric(df["price_eur_mwh"], errors="coerce")

    # MTU parsing:
    # Many ENTSO-E style files use "YYYY-MM-DD HH:MM - YYYY-MM-DD HH:MM"
    # We interpret the *start* as the timestamp for that delivery hour/quarter.
    mtu_str = df["mtu"].astype(str).str.strip()
    start_str = mtu_str.str.split(" - ").str[0]

    # Parse naive local times then localize to Europe/Amsterdam.
    # DST edge cases:
    # - nonexistent times (spring forward): shift forward
    # - ambiguous times (fall back): mark NaT and handle below
    dt_naive = pd.to_datetime(start_str, errors="coerce", dayfirst=False)
    mask_nat = dt_naive.isna()
    if mask_nat.any():
        dt_naive.loc[mask_nat] = pd.to_datetime(start_str.loc[mask_nat], errors="coerce", dayfirst=True)

    # Localize with DST handling
    dt = dt_naive.dt.tz_localize(
        LOCAL_TZ,
        ambiguous="NaT",       # fall-back duplicates become NaT for now
        nonexistent="shift_forward",
    )

    df["ts"] = dt
    df.drop(columns=["mtu"], inplace=True)

    # Remove rows with unparseable timestamps
    df = df.dropna(subset=["ts"]).copy()

    return df


def infer_resolution_minutes(ts: pd.Series) -> int:
    """
    Infer typical resolution from median diff in minutes.
    """
    s = ts.sort_values()
    diffs = s.diff().dropna().dt.total_seconds() / 60.0
    if diffs.empty:
        return -1
    return int(round(diffs.median()))


def make_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> int:
    raw_dir = find_existing_dir([str(Path(p)) for p in RAW_DIR_CANDIDATES])
    processed_dir = Path(PROCESSED_DIR)
    logs_root = Path(LOGS_ROOT)
    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    run_dir = make_run_dir(logs_root)

    files = resolve_expected_files(raw_dir)

    # Ingest
    dfs = []
    file_meta: List[Dict] = []
    for p in files:
        df = read_price_file(p)
        dfs.append(df)

        file_meta.append(
            {
                "file": str(p),
                "sha256": sha256_file(p),
                "rows_read_after_parse": int(len(df)),
            }
        )

    data = pd.concat(dfs, ignore_index=True)

    # Canonical sorting & de-duplication
    data = data.sort_values("ts").reset_index(drop=True)

    # Remove exact duplicates on timestamp (keep last by default; choice is arbitrary but consistent)
    n_before = len(data)
    data = data.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    n_after = len(data)

    # Quality checks
    res_min = infer_resolution_minutes(data["ts"])
    n_missing_price = int(data["price_eur_mwh"].isna().sum())

    # Produce an hourly series for regime learning:
    # - If original already hourly: this does nothing except ensure regularity.
    # - If 15-min (e.g., 2025): downsample to hourly mean (modelling choice).
    #   Alternative could be "take first quarter", but mean is usually safer unless you have market rule reasons.
    hourly = (
        data.set_index("ts")["price_eur_mwh"]
        .resample("1h")
        .mean()
        .to_frame("price_eur_mwh")
        .reset_index()
    )
    hourly["price_eur_mwh"] = hourly["price_eur_mwh"].round(3)

    # Add convenience columns for later splits
    for df_ in (data, hourly):
        df_["date"] = df_["ts"].dt.date
        df_["year"] = df_["ts"].dt.year

    # Save outputs
    out_parquet_raw = processed_dir / "da_prices_harmonised_raw_resolution.parquet"
    out_csv_raw = processed_dir / "da_prices_harmonised_raw_resolution.csv"
    out_parquet_hourly = processed_dir / "da_prices_harmonised_hourly.parquet"
    out_csv_hourly = processed_dir / "da_prices_harmonised_hourly.csv"

    # Parquet preserves timezone cleanly; CSV is for quick inspection.
    data.to_parquet(out_parquet_raw, index=False)
    data.to_csv(out_csv_raw, index=False, encoding="utf-8")

    hourly.to_parquet(out_parquet_hourly, index=False)
    hourly.to_csv(out_csv_hourly, index=False, encoding="utf-8")

    # Write QA + manifest
    qa = {
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "run_dir": str(run_dir),
        "files": file_meta,
        "rows_concat_before_dedup": int(n_before),
        "rows_after_dedup": int(n_after),
        "inferred_resolution_minutes_raw": res_min,
        "missing_prices_raw": n_missing_price,
        "hourly_rows": int(len(hourly)),
        "notes": [
            "MTU parsed as interval start time; localized to Europe/Amsterdam with DST handling.",
            "Ambiguous fall-back timestamps are dropped as NaT during localization; "
            "if this matters, we can implement explicit disambiguation later.",
            "15-min to hourly downsample uses mean; revisit if you want market-specific aggregation logic.",
        ],
        "outputs": {
            "raw_resolution_parquet": str(out_parquet_raw),
            "raw_resolution_csv": str(out_csv_raw),
            "hourly_parquet": str(out_parquet_hourly),
            "hourly_csv": str(out_csv_hourly),
        },
    }

    (run_dir / "01_manifest_ingest_harmonise.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")

    print("=== DA Regimes: Ingest & Harmonise ===")
    print(f"Raw dir          : {raw_dir}")
    print(f"Processed dir    : {processed_dir}")
    print(f"Run dir          : {run_dir}")
    print(f"Resolution (min) : {res_min}")
    print(f"Rows raw         : {n_after} (after dedup, before: {n_before})")
    print(f"Missing prices   : {n_missing_price}")
    print("Wrote:")
    print(f"  - {out_parquet_raw.name}")
    print(f"  - {out_parquet_hourly.name}")
    print(f"Manifest: {run_dir / '01_manifest_ingest_harmonise.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
