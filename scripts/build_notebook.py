"""Write sepsis_early_warning.ipynb from the cells below.

The notebook is the readable, top to bottom version of the project. It calls
the same code in src/ that the scripts use, so there is one implementation.
After writing it, run it so the outputs are saved:

  .venv/bin/python scripts/build_notebook.py
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace sepsis_early_warning.ipynb
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md("""
# Sepsis Early Warning: does it travel?

**This project is a prototype and is not medical advice.**

Veritas AI International High School Hackathon, Health track. Solo entry by Tanner Barber.

**The question.** Can a sepsis early warning model trained at one hospital travel to another? We train on public intensive care data from hospital A, freeze the model and the alert threshold, and evaluate once on hospital B.

**The user.** An intensive care nurse deciding which patient to look at first.

**How to run this notebook.** `pip install -r requirements.txt`, then run every cell from the top. The first run downloads about 40,000 small public files. The two settings searches take about 35 minutes together, so by default this notebook shows their saved results. Set `RERUN_SEARCHES = True` in the first code cell to repeat them.

All of the working code lives in `src/`. This notebook calls it, so there is one implementation, with tests in `tests/`.
""")

code("""
import json, sys, warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
pd.set_option("display.width", 200, "display.max_columns", 40)

RERUN_SEARCHES = False   # True repeats both settings searches (about 35 minutes)
""")

md("""
## 1. The data

**Dataset:** PhysioNet / Computing in Cardiology Challenge 2019, training sets A and B. Public and de-identified. License: Creative Commons Attribution 4.0. Source: https://physionet.org/content/challenge-2019/1.0.0/

**What one row is:** one hour of one patient's stay in an intensive care unit: 7 vital signs, 26 lab values, age, sex, hours in the unit, and a sepsis label.

**The label.** The challenge defines a sepsis time from two things: clinical suspicion of infection (the earlier of antibiotics being ordered and blood cultures being drawn) and organ damage (a two point rise in the SOFA score). The label switches on 6 hours before that time. So the label depends partly on when the staff acted.

The cell below downloads hospital A only. Hospital B is not downloaded or opened until section 8.
""")

code("""
from download import download_hospital
from load import load_hospital

print("new files downloaded for hospital A:", download_hospital("A"))
a = load_hospital("A")
per_patient = a.groupby("pid")["SepsisLabel"].max()
print(f"hospital A: {a.pid.nunique():,} patients, {len(a):,} hourly rows")
print(f"patients who develop sepsis: {int(per_patient.sum()):,} ({100 * per_patient.mean():.1f}%)")
print(f"rows labeled positive: {100 * a.SepsisLabel.mean():.2f}%")
print(f"patients whose label is already on in their first row (late-entry): {int(a.groupby('pid').SepsisLabel.first().sum())}")
""")

code("""
# Reading raw rows: the first hours of the first patient. Most cells are empty.
a[["pid", "ICULOS", "HR", "O2Sat", "Temp", "SBP", "MAP", "Resp", "WBC", "Lactate", "Creatinine", "Age", "Gender", "SepsisLabel"]].head(8)
""")

code("""
# How empty is each column at hospital A? (percent of hourly rows with no value)
empty = (100 * a.drop(columns=["pid", "hosp"]).isna().mean()).round(1)
print(empty.to_string())
""")

md("""
**What we found by reading raw rows.**
- Vital signs are empty in roughly 8% to 15% of hours (temperature 66%). Lab values are empty in 86% to 99.9% of hours. A value being present is itself a signal that something was measured.
- End-tidal CO2 is empty in 100% of hospital A rows, so it is dropped.
- Every sepsis patient has between 6 and 10 positive rows: records stop a few hours after onset.

**Why accuracy is the wrong score here.** About 2% of rows are positive, so a model that never alerts is about 98% accurate and catches nobody. We count per patient instead: caught, missed, and false alarms, reported separately. The exact meanings are in `DEFINITIONS.md`.

## 2. Three piles of patients

Hospital A is cut by patient, never by row: **fit** (the model learns from these), **tune** (used to pick settings), and **dev** (used to set the alert threshold and report hospital A results). If one patient's hours landed in two piles, the model could recognize the patient instead of learning sepsis.
""")

code("""
from split import load_piles, make_piles

piles = load_piles()
assert make_piles(a).equals(piles), "the saved piles should be exactly what the seed produces"
ever = a.groupby("pid")["SepsisLabel"].max().rename("sepsis").reset_index()
piles.merge(ever).groupby("pile").agg(patients=("pid", "size"), sepsis_patients=("sepsis", "sum"), sepsis_share=("sepsis", "mean")).round(4)
""")

md("""
## 3. Inputs, built only from each patient's past

Every input for a patient at hour *t* uses only that patient's rows at hour *t* or earlier. The groups:

| Group | What it holds | First used by |
|---|---|---|
| physiology | latest known value of each vital sign and lab, carried forward | M1 |
| trends | change, mean, lowest and highest of each vital sign over 6 hours | M1 |
| demographics | age and sex | M1 |
| time | hours in the unit, and the gap between hospital and unit admission | M2 |
| measurement | hours since each value was last measured, and counts of labs | M3 |

A lab value appearing in a window is a sign that something was measured. It is not proof a test was ordered at that moment or that a clinician suspected sepsis.
""")

code("""
from features import build_features, feature_groups, model_columns

features = build_features(a)
print({group: len(columns) for group, columns in feature_groups().items()})
for name in ["T0", "M1", "M2", "M3"]:
    print(name, "sees", len(json.loads((ROOT / "results" / "frozen.json").read_text())["models"][name]["columns"]), "inputs")
features[["pid", "ICULOS", "last_HR", "change6_HR", "mean6_HR", "hours_since_WBC", "labs_in_last_6h", "hours_in_unit"]].iloc[20:26]
""")

code("""
# The no-peeking check, on real patients: cut 200 records short and rebuild.
# The inputs for the hours we kept must come out exactly the same.
some = a[a.pid.isin(a.pid.unique()[:200])]
position = some.groupby("pid", sort=False).cumcount()
kept = position < 12
inputs = sum(feature_groups().values(), [])
full = build_features(some)[kept.to_numpy()].reset_index(drop=True)
cut = build_features(some[kept])
pd.testing.assert_frame_equal(full[inputs], cut[inputs], check_dtype=False)
print("passed: removing the future does not change any input")
""")

md("""
## 4. The comparators on hospital A dev

- **Never alert.** Zero false alarms, zero catches.
- **Modified SIRS**, a bedside screening rule: alert when two or more hold: heart rate over 90; temperature over 38 or under 36 degrees C; breathing rate over 20 or PaCO2 under 32; white cell count over 12 or under 4. "Modified" because the full rule also counts immature white cells (bands), which this data does not record.
""")

code("""
from evaluate import evaluate_alerts
from sirs import sirs_alert

features = features.merge(piles, on="pid")
dev = features[features.pile == "dev"].reset_index(drop=True)
sirs_numbers, _ = evaluate_alerts(dev, sirs_alert(dev))
keys = ["sepsis_patients", "caught", "very_early", "missed", "non_sepsis_patients", "false_alarms", "false_alarm_share", "utility"]
pd.Series({k: sirs_numbers[k] for k in keys}, name="Modified SIRS on hospital A dev").round(3).to_frame()
""")

md("""
SIRS fires on most patients. It rarely misses, and it raises a false alarm on about four of every five patients who never develop sepsis.

## 5. The first model

Gradient boosting builds many small decision trees one after another, each fixing mistakes of the ones before. The version used here (scikit-learn's `HistGradientBoostingClassifier`) accepts empty cells directly.

Six candidate settings were written down first, along with the rule for picking one: judge on the tune pile by the share of sepsis patients alerted at or before onset at the 15% alert budget; among candidates within 1 point of the best, keep the smallest model.
""")

code("""
if RERUN_SEARCHES:
    %run scripts/search_settings.py
search = pd.read_csv(ROOT / "results" / "settings_search.csv")
print("tries:", len(search))
search.sort_values("alerted_by_onset", ascending=False).head(8).round(4)
""")

code("""
# The first model on hospital A dev. This was our first look at dev.
first = json.loads((ROOT / "results" / "first_model_a_dev_results.json").read_text())
show = ["caught", "very_early", "missed", "false_alarms", "alerted_by_onset_share", "false_alarm_share", "median_hours_of_warning", "utility", "hourly_auroc"]
order = ["Never alert", "Modified SIRS", "T0", "M1", "M2", "M3"]
pd.DataFrame({k: {c: first[k].get(c) for c in show} for k in order}).T.astype(float).round(3)
""")

md("""
**What the first model showed.**
- Time in the unit alone (T0) scored higher on utility than body signals alone (M1). In this dataset a long stay nearly gives the answer away.
- Most alerts came very early, long before the 12 hour window.
- The settings pick was inside the noise: the tune pile has only 215 sepsis patients.

## 6. The improvement round

Before freezing, we tried to make the model better. The plan and the keep rule were committed first (see `DECISIONS.md`):

- Every choice is made by 3-fold cross-validation over the fit and tune patients together, split by patient. Dev is not used to choose anything.
- A change is kept only if it adds at least 1.5 points of sepsis patients alerted by onset at the 15% budget, and does not lower the utility score by more than 0.01.
""")

code("""
if RERUN_SEARCHES:
    %run scripts/improve.py
log = pd.read_csv(ROOT / "results" / "improvement_log.csv")
log.round(4)
""")

code("""
choice = json.loads((ROOT / "results" / "improvement_choice.json").read_text())
choice
""")

md("""
## 7. The final models and the freeze

The cell below retrains T0, M1, M2 and M3 from scratch with the chosen settings on the fit and tune patients and sets each threshold on dev at the 15% alert budget. It writes to a scratch folder, so the frozen models are not touched. The cell after it checks that the retrained thresholds match the frozen file that was committed before hospital B was opened.

All four models share the same settings. The gap from M2 to M3 shows how much the model leans on what was measured and when. That is a care pattern, not a cause.
""")

code("""
%run scripts/train_and_score_dev.py --check
""")

code("""
frozen = json.loads((ROOT / "results" / "frozen.json").read_text())
candidate = json.loads((ROOT / "data" / "retrain_check" / "a_dev_candidate.json").read_text())
for name, spec in frozen["models"].items():
    same = np.isclose(spec["threshold"], candidate["models"][name]["threshold"], rtol=1e-6)
    print(f"{name:12s} frozen threshold {spec['threshold']:.6f}   retrained here {candidate['models'][name]['threshold']:.6f}   {'match' if same else 'DIFFERENT'}")
print("alert budget:", frozen["budget"], "| settings:", frozen["settings"]["name"], "| trees:", frozen["trees"], "| smoothing hours:", frozen["smooth"])
""")

md("""
## 8. The one hospital B run

**Disclosure: hospital B is not a blind test.** Before any modeling, hospital B was summarized: patient and row counts, sepsis prevalence, positive-row share, stay length, age and sex mix, sepsis rate by age and by sex, and how empty four columns are. Two label-split facts (how often a lactate value is present, and the unit hour of positive rows) were computed on A and B pooled, and they helped motivate two input groups. No model, input selection, or threshold ever used hospital B. So hospital B is a **site-held-out evaluation set, not a blinded test**.

The real run was done once by `scripts/run_frozen_b.py`, which refuses to run unless the frozen file is committed and refuses to run twice. Its numbers are saved in `results/b_results.json`. The cell below repeats the same scoring with the same frozen models and thresholds and checks that it reproduces those saved numbers.

Two rows use hospital B's unlabeled measurements or scores (never its sepsis labels) and are labeled as such.
""")

code("""
from report import evaluate_site, print_table
from rescale import site_norms

print("new files downloaded for hospital B:", download_hospital("B"))
b_raw = load_hospital("B", frozen_run=True)
b_features = build_features(b_raw)
models = {name: joblib.load(ROOT / "models" / f"{name.replace(' ', '_')}.joblib") for name in frozen["models"]}
b_numbers, b_tables, _ = evaluate_site(b_features, frozen, models, site_norms(b_raw))

saved = json.loads((ROOT / "results" / "b_results.json").read_text())
for name in ["M3", "M1", "Modified SIRS"]:
    for key in ["caught", "very_early", "missed", "false_alarms"]:
        assert b_numbers[name][key] == saved[name][key], (name, key)
    assert np.isclose(b_numbers[name]["utility"], saved[name]["utility"], atol=1e-6)
print("reproduced: these numbers match the one frozen run, commit", saved["_run"]["frozen_commit"][:7])
print_table(b_numbers)
""")

md("""
## 9. Results
""")

code("""
a_numbers = json.loads((ROOT / "results" / "a_dev_results.json").read_text())

def column(n):
    r = n.get("ranges", {})
    span = lambda key, pct=True: (f" ({100 * r[key][0]:.0f}% to {100 * r[key][1]:.0f}%)" if pct else f" ({r[key][0]:.3f} to {r[key][1]:.3f})") if key in r else ""
    return {
        "Patients": f"{n['patients']:,}",
        "Sepsis patients": f"{n['sepsis_patients']:,}",
        "Official utility score": f"{n['utility']:.3f}" + span("utility", False),
        "Alerted at or before onset": f"{n['caught'] + n['very_early']} ({100 * n['alerted_by_onset_share']:.0f}%)" + span("alerted_by_onset_share"),
        "  caught (within 12 hours before onset)": f"{n['caught']} ({100 * n['caught_share']:.0f}%)",
        "  very early (more than 12 hours before)": f"{n['very_early']}",
        "Missed (no alert by onset)": f"{n['missed']}",
        "False alarms": f"{n['false_alarms']} of {n['non_sepsis_patients']:,} ({100 * n['false_alarm_share']:.1f}%)" + span("false_alarm_share"),
        "False alerts per true alert": f"{n['false_alerts_per_true_alert']:.1f}",
        "Alerts per 1,000 patient-days": f"{n['alerts_per_1000_patient_days']:.0f}",
        "Median hours of warning": f"{n['median_hours_of_warning']:g}",
        "Hourly AUROC": f"{n['hourly_auroc']:.3f}",
    }

print("The full model (M3), threshold frozen on hospital A dev at the 15% alert budget.")
print("Ranges in brackets are 95% ranges from a bootstrap that resamples patients 1,000 times.")
pd.DataFrame({"Hospital A dev": column(a_numbers["M3"]), "Hospital B": column(b_numbers["M3"])})
""")

code("""
# Every planned model and both comparators, home and away
rows = {}
for key in ["Never alert", "Modified SIRS", "T0", "M1", "M2", "M3", "M3, same share of patients alerted", "M3 rescaled"]:
    x, y = a_numbers[key], b_numbers[key]
    rows[key] = {
        "utility A dev": round(x["utility"], 3), "utility B": round(y["utility"], 3),
        "alerted by onset A dev": f"{100 * x['alerted_by_onset_share']:.0f}%", "alerted by onset B": f"{100 * y['alerted_by_onset_share']:.0f}%",
        "missed B": y["missed"], "false alarms A dev": f"{100 * x['false_alarm_share']:.1f}%", "false alarms B": f"{100 * y['false_alarm_share']:.1f}%",
    }
pd.DataFrame(rows).T
""")

code("""
# Sensitivity at three other alert budgets (thresholds also frozen on hospital A dev)
rows = {}
for key in ["M3 at 5% budget", "M3 at 10% budget", "M3", "M3 at 20% budget"]:
    x, y = a_numbers[key], b_numbers[key]
    rows[key if key != "M3" else "M3 at 15% budget (main)"] = {
        "alerted by onset A dev": f"{100 * x['alerted_by_onset_share']:.0f}%", "false alarms A dev": f"{100 * x['false_alarm_share']:.1f}%",
        "alerted by onset B": f"{100 * y['alerted_by_onset_share']:.0f}%", "false alarms B": f"{100 * y['false_alarm_share']:.1f}%",
    }
pd.DataFrame(rows).T
""")

code("""
# Hours of warning at hospital B, for sepsis patients alerted at or before onset
p = b_tables["M3"]
warned = p[p.outcome.isin(["caught", "very early"])].hours_of_warning
bins = [(0, 2), (3, 5), (6, 8), (9, 12), (13, 24), (25, 48), (49, 10_000)]
labels = ["0 to 2", "3 to 5", "6 to 8", "9 to 12", "13 to 24", "25 to 48", "over 48"]
counts = [int(((warned >= lo) & (warned <= hi)).sum()) for lo, hi in bins]
fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(labels, counts, color="#2450D8")
ax.bar_label(bars)
ax.set_title(f"Hospital B: half of the alerted sepsis patients got {warned.median():g} hours of warning or more", loc="left")
ax.set_xlabel("Hours of warning before onset"); ax.set_ylabel("Patients")
ax.spines[["top", "right"]].set_visible(False)
plt.show()
print(f"Source: hospital B, {len(warned)} sepsis patients with an alert at or before onset. Alerts more than 12 hours early are 'very early', not 'caught'.")
""")

md("""
### Results by age and sex

This is a first check, **not a complete fairness audit**. The data has no race, ethnicity, language or insurance fields, so those groups cannot be checked at all.
""")

code("""
def by_group(patients):
    groups = {
        "Female": patients.male == 0, "Male": patients.male == 1,
        "Under 50": patients.age < 50, "50 to 64": (patients.age >= 50) & (patients.age < 65),
        "65 to 79": (patients.age >= 65) & (patients.age < 80), "80 and over": patients.age >= 80,
    }
    out = {}
    for name, mask in groups.items():
        g = patients[mask]; s = g[g.sepsis]; n = g[~g.sepsis]
        in_time = s.outcome.isin(["caught", "very early"]).sum()
        out[name] = {
            "patients": len(g), "sepsis patients": len(s),
            "alerted by onset": f"{in_time} of {len(s)} ({100 * in_time / max(len(s), 1):.0f}%)",
            "missed": int((s.outcome == "missed").sum()),
            "false alarms": f"{int((n.alerts > 0).sum())} of {len(n)} ({100 * (n.alerts > 0).mean():.1f}%)",
        }
    return pd.DataFrame(out).T

print("Hospital B, full model (M3) at the frozen threshold")
by_group(b_tables["M3"])
""")

md("""
### Late-entry cases

These are sepsis patients whose label is already on in their first recorded hour, so the record holds no history from before the label. They are reported separately.
""")

code("""
pd.DataFrame({
    "Hospital A dev, M3": {**a_numbers["M3"]["late_entry"]}, "Hospital B, M3": {**b_numbers["M3"]["late_entry"]},
    "Hospital B, Modified SIRS": {**b_numbers["Modified SIRS"]["late_entry"]},
})
""")

md("""
## 10. Five failures, read one by one

Chosen by rule from the hospital B run: the first three missed sepsis patients by ID and the first two false alarms by ID, skipping the examples already used in the app. "score/line" is the model's score as a multiple of the alert line, so 1.0 or more means alerting.
""")

code("""
%run scripts/show_failures.py B
""")

md("""
## 11. What this does and does not show

- **Site-held-out, not blind.** See the disclosure in section 8. A truly unseen hospital is future work.
- **Not comparable to the challenge leaderboard.** The challenge scored teams on hidden patients, including a third hospital that was never released. Our test is hospital A to hospital B on public data.
- **Time in the unit explains a lot.** Compare T0 with M1 above. Part of what looks like skill is the shape of this dataset.
- **Very early alerts are not the same as catches.** We report them separately.
- **The label depends on staff behavior.** It is built partly from when antibiotics were ordered and cultures drawn.
- **Scores are relative risk scores, not probabilities.** Calibration was not checked.
- **Retrospective only.** Nothing here was tested with a real nurse or on live data.

Would this be safe to use? **No.** It would need a prospective test at the hospital that would use it, a check that staff act on its alerts, calibration, and a far wider fairness audit.

## Credits

- Data and challenge: Reyna MA, Josef CS, Jeter R, et al. "Early Prediction of Sepsis from Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019." Critical Care Medicine, 2020. Data license CC BY 4.0.
- Scoring: the challenge organizers' `evaluate_sepsis_score.py`, unchanged, BSD 2-Clause license, in `third_party/`.
- Libraries: pandas, NumPy, scikit-learn, Matplotlib, Plotly, Gradio, Jupyter.
- AI tools: Claude Code wrote most of the code from Tanner's instructions and ran the literature checks. Tanner chose the question, the user, the alert budget, and the definitions, adopted the prediction before any model was trained, decided to improve the model before freezing, and read the failures.

This project is a prototype and is not medical advice.
""")

notebook = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
})
nbf.write(notebook, ROOT / "sepsis_early_warning.ipynb")
print("wrote", len(cells), "cells")
