from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

# ---------------------------
# USER-EDITABLE CONFIG
# ---------------------------

EXPECTED_FILES_STEMS = [
    "GUI_ENERGY_PRICES_202112312300-202212312300",
    "GUI_ENERGY_PRICES_202212312300-202312312300",
    "GUI_ENERGY_PRICES_202312312300-202412312300",
    "GUI_ENERGY_PRICES_202412312300-202512312300",
]

REQUIRED_COLUMNS = [
    "MTU (CET/CEST)",
    "Day-ahead Price (EUR/MWh)",
]

# Preferred (matches your current repo layout)
RAW_DIR_CANDIDATES = [
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\1_Configs\1.1_Data_raw\DA-market",
    # Legacy / alternative from your description
    r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\Configs\Data_raw\DA-market",
]

PROCESSED_DIR = r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\1_Configs\1.2_Data_processed\DA-market\00_harmonised"
LOGS_ROOT = r"C:\Users\marijnvalk\PycharmProjects\Optimisation-Model\4_logs\da_regimes"


# ---------------------------
# INTERNALS
# ---------------------------

@dataclass(frozen=True)
class ResolvedPaths:
    raw_dir: str
    processed_dir: str
    run_dir: str


def find_existing_dir(candidates: List[str]) -> Path:
    for c in candidates:
        p = Path(c)
        if p.exists() and p.is_dir():
            return p
    raise FileNotFoundError(
        "None of the RAW_DIR_CANDIDATES exist. "
        "Update RAW_DIR_CANDIDATES at the top of this script."
    )


def resolve_expected_files(raw_dir: Path) -> List[Path]:
    """
    Accepts either exact filenames or filenames with .csv extension.
    """
    found: List[Path] = []
    for stem in EXPECTED_FILES_STEMS:
        exact = raw_dir / stem
        csv = raw_dir / f"{stem}.csv"
        if exact.exists():
            found.append(exact)
        elif csv.exists():
            found.append(csv)
        else:
            found.append(Path())  # placeholder for reporting
    return found


def create_run_dir(logs_root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = logs_root / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main() -> int:
    # Basic environment checks (lightweight by design)
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >= 3.10 is recommended for this pipeline.")

    # Resolve dirs
    raw_dir = find_existing_dir([str(Path(p)) for p in RAW_DIR_CANDIDATES])
    processed_dir = Path(PROCESSED_DIR)
    logs_root = Path(LOGS_ROOT)

    processed_dir.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    run_dir = create_run_dir(logs_root)

    # Resolve files
    expected_files = resolve_expected_files(raw_dir)

    missing = []
    present = []
    for stem, f in zip(EXPECTED_FILES_STEMS, expected_files):
        if not f.exists():
            missing.append(stem)
        else:
            present.append(str(f))

    status = "OK" if not missing else "MISSING_FILES"

    manifest = {
        "status": status,
        "timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "paths": asdict(
            ResolvedPaths(
                raw_dir=str(raw_dir),
                processed_dir=str(processed_dir),
                run_dir=str(run_dir),
            )
        ),
        "expected_files": EXPECTED_FILES_STEMS,
        "present_files": present,
        "missing_files": missing,
        "required_columns": REQUIRED_COLUMNS,
        "note": (
            "This script only checks existence of raw files. "
            "Schema/format validation happens in Script 1."
        ),
    }

    manifest_path = run_dir / "00_manifest_config_check.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Console summary (so PyCharm run output is informative)
    print("=== DA Regimes: Config Check ===")
    print(f"Raw dir       : {raw_dir}")
    print(f"Processed dir : {processed_dir}")
    print(f"Run dir       : {run_dir}")
    print(f"Status        : {status}")
    if missing:
        print("Missing files:")
        for m in missing:
            print(f"  - {m}  (or {m}.csv)")
        return 2

    print("All expected files found.")
    print(f"Manifest written to: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
