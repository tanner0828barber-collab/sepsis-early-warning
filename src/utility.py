"""The challenge's official utility score, using the organizers' own code.

The score looks at every hour, not only the first alert:
  an alert on a sepsis patient earns the most 6 hours before onset,
  earns less the earlier or later it is, and earns nothing after onset + 3;
  staying silent on a sepsis patient near onset costs up to 2 points an hour;
  every alerted hour on a patient who never develops sepsis costs 0.05.

It is scaled so that never alerting scores 0 and perfect alerting scores 1.
Below 0 means worse than never alerting.

We call compute_prediction_utility from the organizers' unchanged script in
third_party/. The loop below copies the normalization steps from their
evaluate_sepsis_score function, which reads files from folders; ours reads
the same labels and predictions from a table instead.
"""
import sys
from pathlib import Path

import numpy as np

THIRD_PARTY = Path(__file__).resolve().parent.parent / "third_party" / "physionet_evaluation_2019"
sys.path.insert(0, str(THIRD_PARTY))

from evaluate_sepsis_score import compute_prediction_utility  # noqa: E402

DT_EARLY, DT_OPTIMAL, DT_LATE = -12, -6, 3


def utility_parts(rows, alert_on):
    """Per-patient utility: what we scored, the best possible, and never alerting.

    Returns three arrays with one entry per patient, in the order patients
    appear in `rows`. Keeping them per patient lets us resample patients later.
    """
    labels_all = rows["SepsisLabel"].to_numpy().astype(int)
    alerts_all = np.asarray(alert_on).astype(int)
    pids = rows["pid"].to_numpy()
    starts = np.flatnonzero(np.r_[True, pids[1:] != pids[:-1]])
    ends = np.r_[starts[1:], len(pids)]

    observed, best, inaction = [], [], []
    for start, end in zip(starts, ends):
        labels = labels_all[start:end]
        alerts = alerts_all[start:end]
        n = len(labels)
        best_alerts = np.zeros(n)
        if labels.any():
            t_sepsis = int(np.argmax(labels)) - DT_OPTIMAL
            best_alerts[max(0, t_sepsis + DT_EARLY): min(t_sepsis + DT_LATE + 1, n)] = 1
        observed.append(compute_prediction_utility(labels, alerts, check_errors=False))
        best.append(compute_prediction_utility(labels, best_alerts, check_errors=False))
        inaction.append(compute_prediction_utility(labels, np.zeros(n), check_errors=False))
    return np.array(observed), np.array(best), np.array(inaction)


def normalize(observed, best, inaction):
    """The organizers' scaling: never alerting is 0, perfect alerting is 1."""
    return float((observed.sum() - inaction.sum()) / (best.sum() - inaction.sum()))


def utility_score(rows, alert_on):
    """Normalized utility for a group of patients.

    `rows` needs pid and SepsisLabel, whole records in time order.
    `alert_on` is True for every hour the model (or rule) is alerting.
    """
    return normalize(*utility_parts(rows, alert_on))
