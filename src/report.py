"""Score one site with frozen models and frozen thresholds.

The same function produces the hospital A dev column and the hospital B
column of the scoreboard, so the two are always computed by identical code.
Nothing in here chooses a threshold from sepsis labels.
"""
import numpy as np
import pandas as pd

from bootstrap import bootstrap_ranges
from evaluate import evaluate_alerts
from metrics import cutoff_alerting_at_most
from rescale import rescale_features
from sirs import sirs_alert
from train import score_rows

PLANNED_MODELS = ["T0", "M1", "M2", "M3"]


def threshold_for_share_of_all(rows, scores, share):
    """Cutoff that alerts on a given share of ALL patients at a site.

    This uses the site's scores but not its sepsis labels, so it can be
    applied at a new hospital without knowing who developed sepsis.
    """
    top = pd.Series(np.asarray(scores)).groupby(rows["pid"].to_numpy(), sort=False).max()
    top = np.sort(top.to_numpy())[::-1]
    allowed = int(np.floor(share * len(top)))
    return cutoff_alerting_at_most(top, allowed)


def evaluate_site(features, frozen, models, site_norms, with_ranges=True):
    """Return (numbers by method, per-patient tables by method, M3 scores)."""
    numbers, tables = {}, {}
    smooth = frozen.get("smooth", 1)

    def record(name, alert_on, scores=None):
        result, patients = evaluate_alerts(features, alert_on, scores)
        if with_ranges:
            result["ranges"] = bootstrap_ranges(patients)
        first = features.groupby("pid", sort=False)[["Age", "Gender"]].first()
        patients["age"], patients["male"] = first["Age"], first["Gender"]
        numbers[name], tables[name] = result, patients

    # the four planned models at their frozen thresholds
    scores = {}
    for name in PLANNED_MODELS:
        spec = frozen["models"][name]
        scores[name] = score_rows(models[name], features, spec["columns"], smooth)
        record(name, scores[name] >= spec["threshold"], scores[name])

    # the main model at three other budgets, thresholds also frozen on hospital A dev
    for budget, threshold in frozen["other_budgets"].items():
        record(f"M3 at {round(float(budget) * 100)}% budget", scores["M3"] >= threshold, scores["M3"])

    # travel attempt 2: alert on the same share of all patients as at hospital A dev
    site_threshold = threshold_for_share_of_all(features, scores["M3"], frozen["share_of_all_patients_alerted"])
    record("M3, same share of patients alerted", scores["M3"] >= site_threshold, scores["M3"])
    numbers["M3, same share of patients alerted"]["threshold_used"] = site_threshold

    # travel attempt 3: values rescaled by this site's own average and spread
    spec = frozen["models"]["M3 rescaled"]
    rescaled = rescale_features(features, site_norms)
    rescaled_scores = score_rows(models["M3 rescaled"], rescaled, spec["columns"], smooth)
    record("M3 rescaled", rescaled_scores >= spec["threshold"], rescaled_scores)

    # the comparators
    record("Modified SIRS", sirs_alert(features))
    record("Never alert", np.zeros(len(features), dtype=bool))

    return numbers, tables, scores["M3"]


def print_table(numbers):
    show = [
        "sepsis_patients", "caught", "very_early", "missed", "false_alarms",
        "false_alarm_share", "alerted_by_onset_share", "median_hours_of_warning",
        "false_alerts_per_true_alert", "alerts_per_1000_patient_days", "utility",
        "hourly_auroc", "hourly_auprc",
    ]
    table = pd.DataFrame({k: {c: v.get(c) for c in show} for k, v in numbers.items()}).T
    with pd.option_context("display.width", 260, "display.max_columns", 30):
        print(table.astype(float).round(3).to_string())
