"""Version 2: 95% ranges and paired comparisons. AFTER-THE-FACT. No new model.

The plan was written in DECISIONS.md before this ran. It reads held-out scores
and cutoffs that earlier version 2 scripts saved, then resamples PATIENTS
1,000 times (seed 2026), the same method as the version 1 ranges.

A paired comparison redraws the same patients for both models, so the range is
for the difference itself. If that range includes zero, this data does not
tell the two models apart.

Writes results/after_the_fact_v2_ranges.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bootstrap import RESAMPLES, SEED, bootstrap_ranges  # noqa: E402
from evaluate import evaluate_alerts  # noqa: E402
from report import threshold_for_share_of_all  # noqa: E402

CAP = 0.25
saved = pd.read_parquet(ROOT / "data" / "v2_held_out_scores.parquet")
cutoffs = json.loads((ROOT / "results" / "after_the_fact_v2_both_hospitals.json").read_text())["cutoffs"]


def stored_cutoff(model, name):
    return {c["outer_fold"]: c["value"] for c in cutoffs if c["model"] == model and c["budget"] == CAP and c["cutoff"] == name}


def alerts(rows, score_column, by_fold):
    on = np.zeros(len(rows), dtype=bool)
    fold = rows.outer_fold.to_numpy()
    for k, cutoff in by_fold.items():
        on[fold == k] = rows[score_column].to_numpy()[fold == k] >= cutoff
    return on


def single(name, where, rows, on):
    numbers, patients = evaluate_alerts(rows, on)
    ranges = bootstrap_ranges(patients)
    out = {"what": name, "scored_on": where}
    for key in ["utility", "caught_share", "alerted_by_onset_share", "false_alarm_share"]:
        out[key] = numbers[key]
        out[key + "_low"], out[key + "_high"] = ranges[key]
    return out, patients


def paired(name, where, new, old):
    """Range of (new minus old), redrawing the same patients for both."""
    assert new.index.equals(old.index)
    rng = np.random.default_rng(SEED)
    n = len(new)

    def parts(p):
        return (p["u_observed"].to_numpy(), p["u_best"].to_numpy(), p["u_inaction"].to_numpy(),
                p["outcome"].isin(["caught", "very early"]).to_numpy(), p["sepsis"].to_numpy(), (p["outcome"] == "false alarm").to_numpy())

    a, b = parts(new), parts(old)

    def numbers(q, pick):
        obs, best, ina, in_time, sepsis, alarm = (v[pick] for v in q)
        return (obs.sum() - ina.sum()) / (best.sum() - ina.sum()), in_time.sum() / sepsis.sum(), alarm.sum() / (~sepsis).sum()

    everyone = np.arange(n)
    point = np.subtract(numbers(a, everyone), numbers(b, everyone))
    draws = np.array([np.subtract(numbers(a, pick), numbers(b, pick)) for pick in (rng.integers(0, n, n) for _ in range(RESAMPLES))])
    low, high = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)
    out = {"what": name, "scored_on": where}
    for i, key in enumerate(["utility", "alerted_by_onset_share", "false_alarm_share"]):
        out[key], out[key + "_low"], out[key + "_high"] = point[i], low[i], high[i]
    return out


rows_a = saved[saved.hosp == "A"].reset_index(drop=True)
rows_b = saved[saved.hosp == "B"].reset_index(drop=True)
both_a_on = alerts(rows_a, "score_both", stored_cutoff("both hospitals", "hospital A"))
both_b_on = alerts(rows_b, "score_both", stored_cutoff("both hospitals", "hospital B"))
only_a_on = alerts(rows_a, "score_A_only", stored_cutoff("A only", "hospital A"))

table = []
row, both_a = single("both-hospitals model", "hospital A", rows_a, both_a_on); table.append(row)
row, both_b = single("both-hospitals model", "hospital B", rows_b, both_b_on); table.append(row)
together = pd.concat([rows_a, rows_b], ignore_index=True)
row, _ = single("both-hospitals model", "both together", together, np.concatenate([both_a_on, both_b_on])); table.append(row)
row, only_a = single("hospital A only model", "hospital A", rows_a, only_a_on); table.append(row)
table.append(paired("DIFFERENCE: both-hospitals minus hospital A only", "hospital A", both_a, only_a))

b_only_file = ROOT / "data" / "v2_held_out_scores_B_only.parquet"
b_only = pd.read_parquet(b_only_file)
if "alert_B_only" in b_only.columns:
    assert (b_only.pid.to_numpy() == rows_b.pid.to_numpy()).all()
    assert (b_only.alert_both.to_numpy() == both_b_on).all()      # the two scripts agree on the baseline's alerts
    row, only_b = single("hospital B only model", "hospital B", rows_b, b_only.alert_B_only.to_numpy()); table.append(row)
    table.append(paired("DIFFERENCE: hospital B only minus both-hospitals", "hospital B", only_b, both_b))
else:
    print("hospital B only alerts not saved yet: rerun scripts/v2_local_only.py first")

# the frozen model at hospital B: its own burn-in alert line against hospital A's cutoff, on the 18,000
line = pd.read_csv(ROOT / "results" / "after_the_fact_v2_alert_line.csv")
frozen_scores = pd.read_parquet(ROOT / "data" / "scores_B_M3.parquet")
burn = set(sorted(frozen_scores.pid.unique())[:2000])
rest = frozen_scores[~frozen_scores.pid.isin(burn)].reset_index(drop=True)
cut_a = float(line.loc[line.alert_line == "hospital A cutoff, unchanged", "cutoff"].iloc[0])
cut_b = float(line.loc[line.alert_line.str.startswith("hospital B's own line"), "cutoff"].iloc[0])
row, unchanged = single("frozen model, hospital A cutoff unchanged", "hospital B, 18,000", rest, rest.score_M3.to_numpy() >= cut_a); table.append(row)
row, own_line = single("frozen model, hospital B's own line from burn-in", "hospital B, 18,000", rest, rest.score_M3.to_numpy() >= cut_b); table.append(row)
table.append(paired("DIFFERENCE: own line minus unchanged cutoff", "hospital B, 18,000", own_line, unchanged))

table = pd.DataFrame(table)
table.to_csv(ROOT / "results" / "after_the_fact_v2_ranges.csv", index=False)
for _, r in table.iterrows():
    diff = r.what.startswith("DIFFERENCE")
    def show(key, pct):
        if pd.isna(r.get(key)): return ""
        scale, unit = (100, " pts" if diff else "%") if pct else (1, "")
        fmt = "+.1f" if (pct and diff) else (".1f" if pct else ("+.3f" if diff else ".3f"))
        return f"{format(r[key] * scale, fmt)}{unit} ({format(r[key + '_low'] * scale, fmt)} to {format(r[key + '_high'] * scale, fmt)})"
    print(f"{r.what} | {r.scored_on}\n    utility {show('utility', False)} | timely {show('caught_share', True)} | by onset {show('alerted_by_onset_share', True)} | false alarms {show('false_alarm_share', True)}")
