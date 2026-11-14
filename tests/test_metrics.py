"""Checks that caught, missed, very early and false alarm are counted as DEFINITIONS.md says."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from sirs import sirs_alert, sirs_count  # noqa: E402


def patient(pid, hours, label_from=None, alert_hours=(), first_hour=1):
    """A made-up patient: label switches on at `label_from`, alerts at `alert_hours`."""
    hour = np.arange(first_hour, first_hour + hours)
    return pd.DataFrame({
        "pid": pid,
        "ICULOS": hour,
        "SepsisLabel": 0 if label_from is None else (hour >= label_from).astype(int),
        "on": np.isin(hour, alert_hours),
    })


def outcomes(*patients):
    rows = pd.concat(patients, ignore_index=True)
    return patient_outcomes(rows, rows["on"])


def test_each_outcome():
    result = outcomes(
        patient("caught", 20, label_from=11, alert_hours=[9, 10]),       # onset 17, warned 8 hours
        patient("at_onset", 20, label_from=11, alert_hours=[17]),        # alert exactly at onset
        patient("very_early", 26, label_from=20, alert_hours=[5]),       # onset 26, warned 21 hours
        patient("never", 20, label_from=11),                              # no alert at all
        patient("too_late", 14, label_from=5, alert_hours=[13]),         # onset 11, alert after it
        patient("false_alarm", 30, alert_hours=[3, 4, 8]),               # two separate alerts
        patient("quiet", 30),
        patient("late_entry", 9, label_from=3, alert_hours=[4], first_hour=3),
    )
    assert result.loc["caught", "outcome"] == "caught"
    assert result.loc["caught", "hours_of_warning"] == 8
    assert result.loc["at_onset", "outcome"] == "caught"
    assert result.loc["at_onset", "hours_of_warning"] == 0
    assert result.loc["very_early", "outcome"] == "very early"
    assert result.loc["never", "outcome"] == "missed"
    assert result.loc["too_late", "outcome"] == "missed"
    assert result.loc["too_late", "alert_after_onset"]
    assert result.loc["false_alarm", "outcome"] == "false alarm"
    assert result.loc["false_alarm", "alerts"] == 2
    assert result.loc["quiet", "outcome"] == "quiet"
    assert result.loc["late_entry", "late_entry"]
    assert not result.loc["caught", "late_entry"]

    summary = summarize(result)
    assert summary["sepsis_patients"] == 6
    assert summary["caught"] == 3            # caught, at_onset, late_entry
    assert summary["very_early"] == 1
    assert summary["missed"] == 2
    assert summary["missed_but_alerted_after_onset"] == 1
    assert summary["non_sepsis_patients"] == 2
    assert summary["false_alarms"] == 1
    assert summary["false_alarm_share"] == 0.5


def test_boundary_of_the_12_hour_window():
    result = outcomes(
        patient("edge_in", 30, label_from=20, alert_hours=[14]),   # onset 26, warned exactly 12
        patient("edge_out", 30, label_from=20, alert_hours=[13]),  # warned 13
    )
    assert result.loc["edge_in", "outcome"] == "caught"
    assert result.loc["edge_out", "outcome"] == "very early"


def test_threshold_alerts_on_at_most_the_budget():
    rows, scores = [], []
    for i in range(10):                      # ten patients who never develop sepsis
        rows.append(patient(f"n{i}", 5))
        scores += [0.01] * 4 + [(i + 1) / 10]
    rows.append(patient("s", 5, label_from=2))
    scores += [0.99] * 5                     # a sepsis patient must not affect the cutoff
    rows = pd.concat(rows, ignore_index=True)
    scores = np.array(scores)

    threshold = threshold_for_budget(rows, scores, budget=0.20)
    assert 0.8 < threshold < 0.9
    result = patient_outcomes(rows, scores >= threshold)
    assert summarize(result)["false_alarms"] == 2


def test_sirs_needs_two_checks_and_ignores_missing_values():
    features = pd.DataFrame({
        "last_HR":    [95.0, 95.0, np.nan, 80.0],
        "last_Temp":  [37.0, 38.5, np.nan, 35.5],
        "last_Resp":  [18.0, 18.0, np.nan, 18.0],
        "last_PaCO2": [40.0, 40.0, np.nan, 30.0],
        "last_WBC":   [8.0, 8.0, np.nan, 8.0],
    })
    assert sirs_count(features).tolist() == [1, 2, 0, 2]
    assert sirs_alert(features).tolist() == [False, True, False, True]


def test_threshold_never_exceeds_the_budget_when_scores_are_tied():
    """A time-only model gives many patients the same top score. Ties must not break the budget."""
    rows, scores = [], []
    for i in range(10):
        rows.append(patient(f"n{i}", 3))
        scores += [0.0, 0.0, 0.9 if i < 1 else 0.5]     # one clear top, nine tied
    rows = pd.concat(rows, ignore_index=True)
    scores = np.array(scores)
    threshold = threshold_for_budget(rows, scores, budget=0.30)   # allows three, but nine are tied
    result = patient_outcomes(rows, scores >= threshold)
    assert summarize(result)["false_alarms"] == 1
