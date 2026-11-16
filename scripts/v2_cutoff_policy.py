"""Sunday night, step 2: alert cutoff rules on the both-hospitals model. AFTER-THE-FACT.

Hospital B was scored once with the frozen version 1 model before this was
written. Nothing here can replace that result. The plan, the rule for keeping
a change, and the guesses were written in DECISIONS.md before this ran.

No model is trained here. It reads the nested scores saved by
scripts/v2_nested_scores.py and tries three ways of turning a score into an
alert. Every rule is chosen on inner-fold patients (scored by a model that
never trained on them) and then applied to outer-fold patients the rule never
saw. Every hospital gets its own cutoff.

  caps        alert on at most 25%, 30%, 35%, 40% or 50% of patients without sepsis
  best cap    try caps from 5% to 60%, keep the one with the highest official
              utility on the inner-fold patients. No cap is promised.
  time bands  at the 25% cap, let the cutoff differ in hours 1 to 12, 13 to 48
              and 49 on. The early and late cutoffs are the middle one times a
              factor. The pair of factors with the highest inner-fold utility wins.

Run with --library hgb (scikit-learn) or --library lgbm (LightGBM).

Writes results/after_the_fact_v2_cutoff_policy_<library>.csv and .json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from metrics import patient_outcomes, summarize, threshold_for_budget  # noqa: E402
from utility import compute_prediction_utility, utility_score  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--library", choices=["hgb", "lgbm"], required=True)
LIBRARY = parser.parse_args().library
ARMS = {"hgb": ["hgb200"], "lgbm": ["lgbm200", "lgbm400"]}[LIBRARY]

OUTER_FOLDS = 3
CAPS = [0.25, 0.30, 0.35, 0.40, 0.50]
SEARCH_CAPS = [round(c, 3) for c in np.arange(0.05, 0.601, 0.025)]
BAND_EDGES = [12, 48]                       # hours 1 to 12, 13 to 48, 49 on
FACTORS = [0.5, 0.7, 1.0, 1.4, 2.0]
BAND_CAP = 0.25
LEAK_LINE = 0.45
started = time.time()


def say(*parts):
    print(f"[{(time.time() - started) / 60:5.1f} min]", *parts, flush=True)


rows = pd.read_parquet(ROOT / "data" / f"v2_nested_{LIBRARY}.parquet")
pid = rows.pid.to_numpy()
hosp = rows.hosp.to_numpy()
outer = rows.outer_fold.to_numpy()
label = rows.SepsisLabel.to_numpy().astype(int)


# ---- a fast copy of the official utility, used only to CHOOSE rules on inner folds.
# Every reported number below comes from the organizers' own function.
def gain_and_best():
    """For every hour: official utility of alerting minus staying silent, and whether
    the hour is inside the window where a perfect alarm would be on."""
    position = rows.groupby("pid", sort=False).cumcount().to_numpy(dtype=float)
    first_label = pd.Series(np.where(label == 1, position, np.nan)).groupby(pid, sort=False).transform("min").to_numpy()
    d = position - (first_label + 6)          # hours after onset. Empty for patients without sepsis
    gain = np.full(len(rows), -0.05)
    early = (d >= -12) & (d <= -6)
    gain[early] = (d[early] + 12) / 6
    late = (d > -6) & (d <= 3)
    gain[late] = (15 + d[late]) / 9
    gain[d > 3] = 0.0                         # after onset plus 3 hours nothing is scored
    return gain, (d >= -12) & (d <= 3)


GAIN, BEST = gain_and_best()


def fast_utility(mask, alert_on):
    return float(GAIN[mask & alert_on].sum() / GAIN[mask & BEST].sum())


def check_fast_utility():
    """The fast copy must agree with the organizers' function on real patients and random alerts."""
    rng = np.random.default_rng(2026)
    sepsis_ids = rows.pid[label == 1].unique()
    other_ids = rows.pid[~rows.pid.isin(sepsis_ids)].unique()
    chosen = set(rng.choice(sepsis_ids, 300, replace=False)) | set(rng.choice(other_ids, 300, replace=False))
    mask = rows.pid.isin(chosen).to_numpy()
    for share in [0.0, 0.1, 0.5, 1.0]:
        alert_on = rng.random(len(rows)) < share
        official = utility_score(rows[mask], alert_on[mask])
        assert abs(official - fast_utility(mask, alert_on)) < 1e-9, (share, official, fast_utility(mask, alert_on))
    # and one patient alone, hour by hour
    one = rows.pid.to_numpy() == sepsis_ids[0]
    labels = label[one]
    silent = compute_prediction_utility(labels, np.zeros(one.sum()), check_errors=False)
    for i in range(int(one.sum())):
        alerts = np.zeros(int(one.sum())); alerts[i] = 1
        assert abs(compute_prediction_utility(labels, alerts, check_errors=False) - silent - GAIN[one][i]) < 1e-9
    say("fast utility matches the organizers' function (600 patients, 4 random alert patterns, 1 patient hour by hour)")


check_fast_utility()

band = np.digitize(rows.ICULOS.to_numpy(), BAND_EDGES, right=True)      # 0 early, 1 middle, 2 late


def measure(mask, alert_on):
    part = rows[mask]
    numbers = summarize(patient_outcomes(part, alert_on[mask]))
    wanted = ["sepsis_patients", "caught", "very_early", "missed", "alerted_by_onset_share", "caught_share",
              "non_sepsis_patients", "false_alarms", "false_alarm_share", "false_alerts_per_true_alert",
              "alerts_per_1000_patient_days"]
    return {"utility": utility_score(part, alert_on[mask]), **{k: numbers[k] for k in wanted}}


table, choices = [], []
for arm in ARMS:
    outer_score = rows[f"{arm}_outer"].to_numpy()
    alerts = {f"cap {c:.0%}": np.zeros(len(rows), dtype=bool) for c in CAPS}
    alerts["best cap, chosen on inner folds"] = np.zeros(len(rows), dtype=bool)
    alerts[f"time bands at the {BAND_CAP:.0%} cap"] = np.zeros(len(rows), dtype=bool)

    for k in range(OUTER_FOLDS):
        inner_score = rows[f"{arm}_inner{k}"].to_numpy()
        for h in ["A", "B"]:
            train = (outer != k) & (hosp == h)
            test = (outer == k) & (hosp == h)
            assert not np.isnan(inner_score[train]).any() and not np.isnan(outer_score[test]).any()
            train_rows = rows[train]

            # fixed caps
            for c in CAPS:
                cutoff = threshold_for_budget(train_rows, inner_score[train], c)
                alerts[f"cap {c:.0%}"][test] = outer_score[test] >= cutoff

            # the cap with the highest utility on the inner-fold patients
            tried = []
            for c in SEARCH_CAPS:
                cutoff = threshold_for_budget(train_rows, inner_score[train], c)
                tried.append((fast_utility(train, inner_score >= cutoff), c, cutoff))
            best_utility, best_cap, cutoff = max(tried)
            alerts["best cap, chosen on inner folds"][test] = outer_score[test] >= cutoff
            choices.append({"arm": arm, "rule": "best cap", "outer_fold": k, "hospital": h,
                            "chosen_cap": best_cap, "inner_utility": best_utility})

            # cutoffs by time in the unit, at the 25% cap
            tried = []
            for early in FACTORS:
                for late in FACTORS:
                    factor = np.array([early, 1.0, late])[band]
                    adjusted = inner_score / factor
                    cutoff = threshold_for_budget(train_rows, adjusted[train], BAND_CAP)
                    # ties go to the plain single cutoff (1, 1), then to factors nearest 1
                    plainness = -abs(np.log(early)) - abs(np.log(late))
                    tried.append((round(fast_utility(train, adjusted >= cutoff), 6), plainness, early, late, cutoff))
            best_utility, _, early, late, cutoff = max(tried)
            factor = np.array([early, 1.0, late])[band]
            alerts[f"time bands at the {BAND_CAP:.0%} cap"][test] = (outer_score / factor)[test] >= cutoff
            choices.append({"arm": arm, "rule": "time bands", "outer_fold": k, "hospital": h,
                            "early_factor": early, "late_factor": late, "inner_utility": best_utility})
        say(arm, "outer fold", k + 1, "rules chosen")

    for rule, alert_on in alerts.items():
        for group, mask in [("hospital A", hosp == "A"), ("hospital B", hosp == "B")]:
            row = {"library": LIBRARY, "arm": arm, "rule": rule, "scored_on": group, **measure(mask, alert_on)}
            table.append(row)
            flag = "  <-- ABOVE THE LEAK LINE, CHECK BEFORE REPORTING" if row["utility"] > LEAK_LINE else ""
            say(arm, "|", rule, "|", group, "|", f'utility {row["utility"]:.3f}',
                f'timely {row["caught"]}', f'by onset {row["alerted_by_onset_share"]:.1%}',
                f'missed {row["missed"]} of {row["sepsis_patients"]}',
                f'false alarms {row["false_alarm_share"]:.1%}' + flag)

results = pd.DataFrame(table)
results.to_csv(ROOT / "results" / f"after_the_fact_v2_cutoff_policy_{LIBRARY}.csv", index=False)
(ROOT / "results" / f"after_the_fact_v2_cutoff_policy_{LIBRARY}.json").write_text(json.dumps({
    "label": "AFTER-THE-FACT. Hospital B had already been scored once with the frozen version 1 model.",
    "design": {
        "model": "both hospitals, nested, 3 outer folds and 2 inner folds, same folds as Saturday",
        "rules": "each rule is chosen on inner-fold scores inside each outer training set, per hospital",
        "caps": CAPS, "search_caps": SEARCH_CAPS, "band_edges_hours": BAND_EDGES, "factors": FACTORS,
    },
    "results": table,
    "choices": choices,
}, indent=2))

if LIBRARY == "hgb":
    # the 25% cap must be Saturday's result, or something here is different
    saturday = pd.read_csv(ROOT / "results" / "after_the_fact_v2_both_hospitals.csv")
    saturday = saturday[(saturday.model == "both hospitals") & (saturday.cutoff == "per hospital") & (saturday.cap == 0.25)]
    for group in ["hospital A", "hospital B"]:
        then = saturday[saturday.scored_on == group].iloc[0]
        now = results[(results.rule == "cap 25%") & (results.scored_on == group)].iloc[0]
        assert abs(then.utility - now.utility) < 1e-9 and then.missed == now.missed, (group, then.utility, now.utility)
    say("the 25% cap reproduces Saturday's both-hospitals result exactly")
say("wrote results/" + f"after_the_fact_v2_cutoff_policy_{LIBRARY}.csv and .json")
