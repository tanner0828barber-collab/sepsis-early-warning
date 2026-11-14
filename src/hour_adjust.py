"""Grade each score against other patients at the same hour of their stay.

The model's raw score climbs with time in the unit. One fixed alert line is
then rarely crossed in a patient's first day. This turns a raw score into
"the share of all patients at this same hour who score lower," a number from
0 to 1. It uses no sepsis labels.

The table of score levels per hour is built once from hospital A and frozen.
Hours after the second day are pooled into wider bins, because few patients
stay that long.
"""
import numpy as np

# hour bins: every hour up to 48, then wider bins
EDGES = np.array(list(range(1, 49)) + [60, 72, 96, 10_000])
LEVELS = np.linspace(0, 1, 201)   # the 0th, 0.5th, 1st ... 100th percentile


def hour_bin(hours):
    """Which bin each hour falls in. Hours 1 to 48 get their own bin."""
    return np.searchsorted(EDGES, np.asarray(hours), side="left")


def fit_hour_table(hours, scores):
    """For each hour bin, the score at each percentile level. Shape: bins x 201."""
    bins = hour_bin(hours)
    scores = np.asarray(scores)
    table = np.zeros((len(EDGES), len(LEVELS)))
    for b in range(len(EDGES)):
        in_bin = scores[bins == b]
        table[b] = np.quantile(in_bin, LEVELS) if len(in_bin) else np.nan
    return table


def hour_adjusted(hours, scores, table):
    """Each score as a percentile (0 to 1) among patients at the same hour."""
    bins = np.clip(hour_bin(hours), 0, len(EDGES) - 1)
    scores = np.asarray(scores)
    out = np.zeros(len(scores))
    for b in np.unique(bins):
        here = bins == b
        out[here] = np.interp(scores[here], table[b], LEVELS)
    return out
