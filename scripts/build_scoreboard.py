"""Turn the frozen result files into the numbers and sentences the app shows.

Every number on screen comes from results/a_dev_results.json and
results/b_results.json, written by the scoring scripts. Nothing is typed in
by hand, so the screen cannot drift from the notebook.

Before the hospital B run this builds a development preview from hospital A
dev only, and says so.
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

a = json.loads((RESULTS / "a_dev_results.json").read_text())
b_file = RESULTS / "b_results.json"
b = json.loads(b_file.read_text()) if b_file.exists() else None
frozen_file = RESULTS / "frozen.json"
frozen = json.loads(frozen_file.read_text()) if frozen_file.exists() else json.loads((RESULTS / "a_dev_candidate.json").read_text())
budget = round(frozen["budget"] * 100)


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def with_range(numbers, key, kind="pct"):
    """A number with its 95% range in gray underneath."""
    value = numbers[key]
    text = pct(value) if kind == "pct" else f"{value:.3f}"
    if "ranges" in numbers and key in numbers["ranges"]:
        low, high = numbers["ranges"][key]
        span = f"{pct(low)} to {pct(high)}" if kind == "pct" else f"{low:.3f} to {high:.3f}"
        text += f'<br><span class="range">{span}</span>'
    return text


def site_column(numbers):
    """The rows of the two-hospital scoreboard for one site."""
    return [
        with_range(numbers, "utility", "num"),
        f'{numbers["caught"]} of {numbers["sepsis_patients"]}<br><span class="range">{pct(numbers["caught_share"])}</span>',
        f'{numbers["very_early"]} of {numbers["sepsis_patients"]}',
        f'{numbers["missed"]} of {numbers["sepsis_patients"]}',
        f'{numbers["false_alarms"]} of {numbers["non_sepsis_patients"]}<br>' + with_range(numbers, "false_alarm_share").replace("<br>", " "),
        f'{numbers["alerts_per_1000_patient_days"]:.0f}',
        f'{numbers["median_hours_of_warning"]:g} hours',
    ]


ROW_NAMES = [
    "Official utility score (0 is never alerting, 1 is perfect)",
    "Sepsis patients caught (first alert in the 12 hours before onset)",
    "Alerted very early (more than 12 hours before onset)",
    "Sepsis patients missed (no alert by onset)",
    "False alarms (patients who never developed sepsis, alerted at least once)",
    "Alerts per 1,000 patient-days",
    "Median hours of warning (patients alerted by onset)",
]

main_a = a["M3"]
out = {}

if b is None:
    site, main = "Hospital A dev (development preview)", main_a
    two = {"columns": ["Full model (M3)", "Hospital A dev"], "rows": [[n, v] for n, v in zip(ROW_NAMES, site_column(main_a))]}
    out["headline"] = "Development preview. Hospital B has not been run."
    out["finding"] = "These numbers are from hospital A dev patients only. The hospital B column appears after the one frozen run."
    models_source, models_site = a, "hospital A dev"
else:
    site, main = "Hospital B", b["M3"]
    two = {
        "columns": ["Full model (M3)", "Hospital A dev", "Hospital B"],
        "rows": [[n, x, y] for n, x, y in zip(ROW_NAMES, site_column(main_a), site_column(main))],
    }
    direction = "fell" if main["utility"] < main_a["utility"] else "rose"
    out["headline"] = "Trained at hospital A, frozen, then tested once at hospital B."
    out["finding"] = (
        f'The official utility score {direction} from {main_a["utility"]:.3f} at hospital A dev to {main["utility"]:.3f} at hospital B. '
        f'At hospital B the model alerted by onset on {main["caught"] + main["very_early"]} of {main["sepsis_patients"]} sepsis patients '
        f'({pct(main["alerted_by_onset_share"])}, against {pct(main_a["alerted_by_onset_share"])} at A dev) and missed {main["missed"]}. '
        f'It raised a false alarm on {main["false_alarms"]} of {main["non_sepsis_patients"]} patients who never developed sepsis '
        f'({pct(main["false_alarm_share"], 1)}, against the {budget}% budget set at hospital A).'
    )
    models_source, models_site = b, "hospital B"

two["caption"] = (
    f"The threshold was set once on hospital A dev at a {budget}% alert budget and never changed. "
    "Gray ranges are 95% ranges from a bootstrap that resamples patients 1,000 times. "
    f'Hospital A dev is {main_a["patients"]:,} patients the model did not train on. '
    + (f'Hospital B is {b["M3"]["patients"]:,} patients from a different hospital, scored once.' if b else "")
)
out["two_hospitals"] = two

# every planned model and both comparators
names = [
    ("Never alert", "Never alert"),
    ("Modified SIRS", "Modified SIRS rule (two or more of four checks)"),
    ("T0", "T0: hours in the unit only"),
    ("M1", "M1: body signals and trends"),
    ("M2", "M2: M1 plus time"),
    ("M3", "M3: M2 plus measurement signals (the full model)"),
]


def model_row(label, n, home=None):
    row = [
        label,
        f'{n["utility"]:.3f}',
        f'{n["caught"] + n["very_early"]} of {n["sepsis_patients"]} ({pct(n["alerted_by_onset_share"])})',
        f'{n["missed"]}',
        f'{n["false_alarms"]} of {n["non_sepsis_patients"]} ({pct(n["false_alarm_share"], 1)})',
    ]
    if home is not None:
        row.insert(1, f'{home["utility"]:.3f}')
    return row


if b is None:
    out["models"] = {
        "columns": ["Method", "Utility", "Alerted by onset", "Missed", "False alarms"],
        "rows": [model_row(label, a[key]) for key, label in names],
        "caption": f"All on hospital A dev. Each model has its own threshold, set at the same {budget}% budget.",
    }
else:
    out["models"] = {
        "columns": ["Method", "Utility at A dev", "Utility at B", "Alerted by onset at B", "Missed at B", "False alarms at B"],
        "rows": [model_row(label, b[key], a[key]) for key, label in names],
        "caption": (
            f"Each model's threshold was set on hospital A dev at the same {budget}% budget, then applied unchanged to hospital B. "
            "The gap between M2 and M3 shows how much the model leans on what was measured and when. That is a care pattern, not a cause."
        ),
    }
    attempts = [
        ("M3", "Main result: frozen score cutoff from hospital A"),
        ("M3, same share of patients alerted", "Label-free adjustment: alert on the same share of all patients as at A dev"),
        ("M3 rescaled", "Label-free adjustment: values rescaled by each hospital's own average and spread"),
    ]
    out["travel_attempts"] = {
        "columns": ["Attempt", "Utility at A dev", "Utility at B", "Alerted by onset at B", "Missed at B", "False alarms at B"],
        "rows": [model_row(label, b[key], a[key]) for key, label in attempts],
        "caption": (
            "The two adjustments use hospital B's measurements and scores but never its sepsis labels. "
            "They were planned before the hospital B run and are reported whether or not they helped."
        ),
    }

# the proof strip: three numbers, each with its denominator and its site
in_time = main["caught"] + main["very_early"]
out["proof"] = [
    {"big": f'{in_time} of {main["sepsis_patients"]}',
     "what": "sepsis patients alerted at or before onset",
     "where": f'{site}, {main["patients"]:,} patients it never trained on'},
    {"big": f'{main["median_hours_of_warning"]:g} hours',
     "what": "median warning before onset",
     "where": f"{site}, among the {in_time} patients alerted by onset"},
    {"big": pct(main["false_alarm_share"], 1),
     "what": "of patients who never developed sepsis got an alert",
     "where": f'{site}: {main["false_alarms"]:,} of {main["non_sepsis_patients"]:,} patients'},
]

# hours of warning, for patients alerted by onset
patients_file = RESULTS / ("b_patients_M3.csv" if b else "a_dev_patients_M3.csv")
patients = pd.read_csv(patients_file)
warned = patients[patients.outcome.isin(["caught", "very early"])].hours_of_warning
bins = [(0, 2), (3, 5), (6, 8), (9, 12), (13, 24), (25, 48), (49, 10_000)]
labels = ["0 to 2", "3 to 5", "6 to 8", "9 to 12", "13 to 24", "25 to 48", "over 48"]
out["warning_chart"] = {
    "labels": labels,
    "counts": [int(((warned >= lo) & (warned <= hi)).sum()) for lo, hi in bins],
    "title": f"Half of the alerted sepsis patients got {warned.median():g} hours of warning or more",
    "source": (
        f"Source: {site}, {len(warned)} sepsis patients with an alert at or before onset. "
        "Onset is the challenge's sepsis time. Alerts more than 12 hours early are counted separately as very early."
    ),
}

decisions = (ROOT / "DECISIONS.md").read_text()
happened = "Hospital B has not been run yet." if b is None else out["finding"]
out["prediction_box"] = (
    "<b>Before running this, the prediction was:</b> the utility score goes down at hospital B, false alarms go above the "
    f"{budget}% budget (low confidence), and the full model M3 loses more than the body-signals model M1 when it travels. "
    "Committed on Sat 11/14/2026 at 2:20 PM Phoenix time, before any model was trained (first commit in the repository)."
    f"<br><br><b>What happened:</b> {happened}"
)

target = ROOT / "app" / "data"
target.mkdir(parents=True, exist_ok=True)
(target / "scoreboard.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out["proof"], indent=2))
print(out["finding"])
