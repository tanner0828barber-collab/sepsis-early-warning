"""Pick the 12 patients for the app's Unit board tab. By rule, not by eye.

The rule, written in DECISIONS.md before this ran:

  one seeded random draw (seed 2026) from all 20,000 hospital B patients:
  9 who never develop sepsis and 3 who do.

This script never loads the model and never reads a score or an outcome, so
nobody chose patients the model does well or badly on. Bed numbers are made
up and come from the same seeded shuffle, so a bed number says nothing about
the patient.

3 of 12 is far more sepsis than the real share at hospital B (about 1 in 18).
The board says so on screen.

Writes app/data/unit_patients.csv (the 12 public, de-identified records) and
app/data/unit.json (which record sits in which bed).

Run it:  .venv/bin/python scripts/build_unit_board.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from load import load_hospital  # noqa: E402

SEED = 2026
WITHOUT_SEPSIS, WITH_SEPSIS = 9, 3

# Hospital B was locked until the one frozen run. That run is done and committed, so opening it is allowed.
raw = load_hospital("B", frozen_run=True)
ever_sepsis = raw.groupby("pid", sort=True)["SepsisLabel"].max()

rng = np.random.default_rng(SEED)
without = rng.choice(ever_sepsis.index[ever_sepsis == 0].to_numpy(), WITHOUT_SEPSIS, replace=False)
with_sepsis = rng.choice(ever_sepsis.index[ever_sepsis == 1].to_numpy(), WITH_SEPSIS, replace=False)
in_bed_order = rng.permutation(np.concatenate([without, with_sepsis]))

out = ROOT / "app" / "data"
keep = raw[raw.pid.isin(set(in_bed_order))].drop(columns=["hosp"])
keep.to_csv(out / "unit_patients.csv", index=False)
(out / "unit.json").write_text(json.dumps({
    "site": "Hospital B",
    "rule": (
        f"one seeded random draw (seed {SEED}) from all {len(ever_sepsis):,} hospital B patients: "
        f"{WITHOUT_SEPSIS} who never develop sepsis and {WITH_SEPSIS} who do. The draw never reads the model's scores"
    ),
    "seed": SEED,
    "with_sepsis": WITH_SEPSIS,
    "hospital_sepsis_patients": int(ever_sepsis.sum()),
    "hospital_patients": int(len(ever_sepsis)),
    "beds": [{"bed": number, "pid": pid} for number, pid in enumerate(in_bed_order.tolist(), start=1)],
}, indent=2))
print("beds:", len(in_bed_order), "rows saved:", len(keep))
