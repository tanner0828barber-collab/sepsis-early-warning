"""Checks that our wrapper around the organizers' scoring code is scaled correctly."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utility import utility_score  # noqa: E402


def make_rows():
    """Two sepsis patients and three who never develop sepsis."""
    tables = []
    for i, label_from in enumerate([20, 15, None, None, None]):
        hour = np.arange(1, 31)
        label = np.zeros(30, dtype=int) if label_from is None else (hour >= label_from).astype(int)
        tables.append(pd.DataFrame({"pid": f"p{i}", "ICULOS": hour, "SepsisLabel": label}))
    return pd.concat(tables, ignore_index=True)


def test_never_alerting_scores_zero():
    rows = make_rows()
    assert utility_score(rows, np.zeros(len(rows), dtype=bool)) == 0.0


def test_perfect_alerting_scores_one():
    rows = make_rows()
    perfect = np.zeros(len(rows), dtype=bool)
    for pid, label_from in [("p0", 20), ("p1", 15)]:
        onset = label_from + 6
        window = (rows.pid == pid) & (rows.ICULOS >= onset - 12) & (rows.ICULOS <= onset + 3)
        perfect |= window.to_numpy()
    assert abs(utility_score(rows, perfect) - 1.0) < 1e-9


def test_alerting_only_on_healthy_patients_is_below_zero():
    rows = make_rows()
    wrong = rows.pid.isin(["p2", "p3", "p4"]).to_numpy()
    assert utility_score(rows, wrong) < 0
