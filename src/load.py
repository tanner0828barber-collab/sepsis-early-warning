"""Load the PhysioNet 2019 challenge files into one table per hospital.

Each patient is one small text file with one row per hour in the unit.
This stacks a hospital's files into a single table and adds two columns:
  pid   the patient's file name, for example "p000001"
  hosp  "A" or "B"

Hospital B is locked. It can only be loaded by passing frozen_run=True,
which happens exactly once, after the threshold is frozen and committed.
"""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HOSPITAL_FOLDERS = {"A": "training_setA", "B": "training_setB"}


def load_hospital(hospital, data_dir=DATA_DIR, frozen_run=False, use_cache=True):
    """Return every hourly row for one hospital as a pandas table."""
    if hospital not in HOSPITAL_FOLDERS:
        raise ValueError("hospital must be 'A' or 'B'")
    if hospital == "B" and not frozen_run:
        raise PermissionError(
            "Hospital B is locked until the one frozen run. "
            "All development uses hospital A only."
        )

    data_dir = Path(data_dir)
    cache_file = data_dir / f"hospital_{hospital}.parquet"
    if use_cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    folder = data_dir / HOSPITAL_FOLDERS[hospital]
    files = sorted(folder.glob("p*.psv"))
    if not files:
        raise FileNotFoundError(
            f"No patient files found in {folder}. Run the download cell first."
        )

    tables = []
    for path in files:
        one_patient = pd.read_csv(path, sep="|")
        one_patient.insert(0, "pid", path.stem)
        tables.append(one_patient)

    table = pd.concat(tables, ignore_index=True)
    table.insert(1, "hosp", hospital)
    if use_cache:
        table.to_parquet(cache_file)
    return table
