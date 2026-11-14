"""Ranges for the reported numbers, by resampling patients.

A bootstrap asks: if we had drawn a different group of patients from the same
hospital, how much would this number move? We redraw the patient list 1,000
times, with replacement, and report the middle 95% of the results.

We resample patients, never rows. One patient's hours are not independent of
each other, so resampling rows would make the ranges look far too narrow.
"""
import numpy as np

RESAMPLES = 1000
SEED = 2026


def bootstrap_ranges(patients, resamples=RESAMPLES, seed=SEED):
    """Return {number name: (low, high)} for one per-patient table from evaluate_alerts."""
    rng = np.random.default_rng(seed)
    sepsis = patients["sepsis"].to_numpy()
    caught = (patients["outcome"] == "caught").to_numpy()
    in_time = patients["outcome"].isin(["caught", "very early"]).to_numpy()
    false_alarm = (patients["outcome"] == "false alarm").to_numpy()
    observed = patients["u_observed"].to_numpy()
    best = patients["u_best"].to_numpy()
    inaction = patients["u_inaction"].to_numpy()

    draws = {"caught_share": [], "alerted_by_onset_share": [], "false_alarm_share": [], "utility": []}
    n = len(patients)
    for _ in range(resamples):
        pick = rng.integers(0, n, n)
        n_sepsis = sepsis[pick].sum()
        n_other = n - n_sepsis
        draws["caught_share"].append(caught[pick].sum() / n_sepsis)
        draws["alerted_by_onset_share"].append(in_time[pick].sum() / n_sepsis)
        draws["false_alarm_share"].append(false_alarm[pick].sum() / n_other)
        draws["utility"].append(
            (observed[pick].sum() - inaction[pick].sum()) / (best[pick].sum() - inaction[pick].sum())
        )
    return {
        name: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for name, values in draws.items()
    }
