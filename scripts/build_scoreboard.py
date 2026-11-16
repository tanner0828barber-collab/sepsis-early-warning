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


def pct(x, digits=1):
    """One percent style everywhere on screen: one decimal."""
    return f"{100 * x:.{digits}f}%"


def n(x):
    """A count with commas."""
    return f"{int(x):,}"


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
    sepsis = numbers["sepsis_patients"]
    return [
        with_range(numbers, "utility", "num"),
        f'{n(numbers["caught"])} of {n(sepsis)}<br><span class="range">{pct(numbers["caught_share"])}</span>',
        f'{n(numbers["very_early"])} of {n(sepsis)}<br><span class="range">{pct(numbers["very_early"] / sepsis)}</span>',
        f'{n(numbers["missed"])} of {n(sepsis)}<br><span class="range">{pct(numbers["missed"] / sepsis)}</span>',
        f'{n(numbers["false_alarms"])} of {n(numbers["non_sepsis_patients"])}<br>' + with_range(numbers, "false_alarm_share").replace("<br>", " "),
        f'{numbers["false_alarms"] / numbers["caught"]:.1f}',
        f'{numbers["alerts_per_1000_patient_days"]:.0f}',
        f'{numbers["median_hours_of_warning"]:g} hours',
    ]


ROW_NAMES = [
    "Official utility score (0 is never alerting, 1 is perfect)",
    "Timely catches (first alert in the 12 hours before onset)",
    "Alerted very early (more than 12 hours before onset)",
    "Sepsis patients missed (no alert by onset)",
    "False alarms (patients who never developed sepsis, alerted at least once)",
    "False alarms per timely catch",
    "Alerts per 1,000 patient-days",
    "Median hours of warning (all patients alerted by onset, most of them very early)",
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
        "columns": ["Full model (M3)", "Hospital A dev (calibration set)", "Hospital B"],
        "rows": [[n, x, y] for n, x, y in zip(ROW_NAMES, site_column(main_a), site_column(main))],
    }
    direction = "fell" if main["utility"] < main_a["utility"] else "rose"
    out["headline"] = "Trained at hospital A, frozen, then tested once at hospital B."
    out["finding"] = (
        f'At hospital B, {n(main["caught"])} of {n(main["sepsis_patients"])} sepsis patients ({pct(main["caught_share"])}) got their first alert '
        f'in the 12 hours before onset. Another {n(main["very_early"])} were alerted earlier than that, and {n(main["missed"])} were missed. '
        f'It raised a false alarm on {n(main["false_alarms"])} of {n(main["non_sepsis_patients"])} patients who never developed sepsis '
        f'({pct(main["false_alarm_share"])}, against the {budget}% budget set at hospital A). '
        f'The official utility score {direction} from {main_a["utility"]:.3f} at hospital A dev (the calibration set) to {main["utility"]:.3f} at hospital B. '
        f'A model that sees only hours in the unit scored {b["T0"]["utility"]:.3f} there.'
    )
    models_source, models_site = b, "hospital B"

two["caption"] = (
    f"The threshold was set once on hospital A dev at a {budget}% alert budget and never changed. "
    f"That makes hospital A dev the calibration set: its {budget}% false alarm share is true by construction, not a measured result. "
    "Gray ranges are 95% ranges from a bootstrap that resamples patients 1,000 times. "
    f'Hospital A dev is {main_a["patients"]:,} patients the model did not train on. '
    + (f'Hospital B is {b["M3"]["patients"]:,} patients from a different hospital, scored once.' if b else "")
)
out["two_hospitals"] = two

# The result strip at the top of every tab, and one mark per sepsis patient for the dot picture.
# Same frozen numbers as the table above, nothing new is computed here.
if b is not None:
    sepsis_b, sepsis_a = main["sepsis_patients"], main_a["sepsis_patients"]
    out["strip"] = [
        {"big": f'{n(main["caught"])} of {n(sepsis_b)}', "what": "sepsis patients got a timely alert, in the 12 hours before onset"},
        {"big": f'{n(main["missed"])} of {n(sepsis_b)}', "what": "sepsis patients were missed: no alert by onset"},
        {"big": f'{n(main["false_alarms"])} of {n(main["non_sepsis_patients"])}',
         "what": f'patients who never developed sepsis got a false alarm ({pct(main["false_alarm_share"])})'},
    ]
    out["dots"] = {
        "title": (
            f'Missed sepsis patients: {pct(main_a["missed"] / sepsis_a, 0)} at home, '
            f'{pct(main["missed"] / sepsis_b, 0)} at a second hospital'
        ),
        "sites": [
            {"name": name, "sepsis": int(x["sepsis_patients"]), "timely": int(x["caught"]), "very_early": int(x["very_early"]), "missed": int(x["missed"])}
            for name, x in [("Hospital A dev (the calibration set, at home)", main_a), ("Hospital B (never trained on)", main)]
        ],
        "caption": (
            "One mark per sepsis patient. Trained at hospital A, frozen, then scored once at hospital B. Timely means the first alert came in "
            "the 12 hours before onset. Very early means more than 12 hours before onset. Missed means no alert by onset. "
            "Source: the frozen result files, the same numbers as the table below."
        ),
    }

# every planned model and both comparators
names = [
    ("Never alert", "Never alert"),
    ("Modified SIRS", "Modified SIRS rule (two or more of four checks)"),
    ("T0", "T0: hours in the unit only"),
    ("M1", "M1: body signals and trends"),
    ("M2", "M2: M1 plus time"),
    ("M3", "M3: M2 plus measurement signals (the full model)"),
]


def model_row(label, m, home=None):
    row = [
        label,
        f'{m["utility"]:.3f}',
        f'{n(m["caught"])}',
        f'{n(m["caught"] + m["very_early"])} of {n(m["sepsis_patients"])} ({pct(m["alerted_by_onset_share"])})',
        f'{n(m["missed"])}',
        f'{n(m["false_alarms"])} of {n(m["non_sepsis_patients"])} ({pct(m["false_alarm_share"])})',
    ]
    if home is not None:
        row.insert(1, f'{home["utility"]:.3f}')
    return row


if b is None:
    out["models"] = {
        "columns": ["Method", "Utility", "Timely", "Alerted by onset", "Missed", "False alarms"],
        "rows": [model_row(label, a[key]) for key, label in names],
        "caption": f"All on hospital A dev. Each model has its own threshold, set at the same {budget}% budget.",
    }
else:
    out["models"] = {
        "columns": ["Method", "Utility at A dev (calibration set)", "Utility at B", "Timely at B", "Alerted by onset at B", "Missed at B", "False alarms at B"],
        "rows": [model_row(label, b[key], a[key]) for key, label in names],
        "caption": (
            f"Each model's threshold was set on hospital A dev at the same {budget}% budget, then applied unchanged to hospital B. "
            '"Timely" means the first alert came in the 12 hours before onset. '
            "The gap between M2 and M3 shows how much the model leans on what was measured and when. That is a care pattern, not a cause."
        ),
    }
    attempts = [
        ("M3", "Main result: frozen score cutoff from hospital A"),
        ("M3, same share of patients alerted", "Label-free adjustment: alert on the same share of all patients as at A dev"),
        ("M3 rescaled", "Label-free adjustment: values rescaled by each hospital's own average and spread"),
    ]
    out["travel_attempts"] = {
        "columns": ["Attempt", "Utility at A dev (calibration set)", "Utility at B", "Timely at B", "Alerted by onset at B", "Missed at B", "False alarms at B"],
        "rows": [model_row(label, b[key], a[key]) for key, label in attempts],
        "caption": (
            "The two adjustments use hospital B's measurements and scores but never its sepsis labels. "
            "They were planned before the hospital B run and are reported whether or not they helped."
        ),
    }

# the proof tile: the timely number leads, then alerted by onset, missed, and false alarms
in_time = main["caught"] + main["very_early"]
out["proof"] = {
    "title": f'{site}, {main["patients"]:,} patients it never trained on',
    "lede": out["headline"],
    "numbers": [
        {"big": f'{n(main["caught"])} of {n(main["sepsis_patients"])}',
         "what": "sepsis patients got a timely alert, in the 12 hours before onset"},
        {"big": n(in_time), "what": "alerted by onset, however early"},
        {"big": n(main["missed"]), "what": "missed"},
        {"big": pct(main["false_alarm_share"]), "what": "false alarms, among patients who never developed sepsis"},
    ],
}

# hours of warning, for patients alerted by onset
patients_file = RESULTS / ("b_patients_M3.csv" if b else "a_dev_patients_M3.csv")
patients = pd.read_csv(patients_file)
warned = patients[patients.outcome.isin(["caught", "very early"])].hours_of_warning
bins = [(0, 2), (3, 5), (6, 8), (9, 12), (13, 24), (25, 48), (49, 10_000)]
labels = ["0 to 2", "3 to 5", "6 to 8", "9 to 12", "13 to 24", "25 to 48", "over 48"]
out["warning_chart"] = {
    "labels": labels,
    "counts": [int(((warned >= lo) & (warned <= hi)).sum()) for lo, hi in bins],
    "timely": [hi <= 12 for lo, hi in bins],   # which bars sit inside the 12 hour window
    "title": f"Most alerts came more than 12 hours early: {n((warned > 12).sum())} of the {n(len(warned))} patients alerted by onset",
    "source": (
        f"Source: {site}, {n(len(warned))} sepsis patients with an alert at or before onset. "
        "Onset is the challenge's sepsis time. Alerts more than 12 hours early are counted separately as very early."
    ),
}

# the "Known weak spots" section of the limits card, written from the result files
sepsis = patients[patients.sepsis]
first_day, later = sepsis[sepsis.label_start < 24], sepsis[sepsis.label_start >= 24]


def by_onset(group):
    return int(group.outcome.isin(["caught", "very early"]).sum())


def group_row(name, mask):
    g = patients[mask]
    s, others = g[g.sepsis], g[~g.sepsis]
    timely, missed = int((s.outcome == "caught").sum()), int((s.outcome == "missed").sum())
    alarms = int((others.outcome == "false alarm").sum())
    return (
        f"| {name} | {n(len(s))} | {n(timely)} ({pct(timely / len(s))}) | {n(by_onset(s))} ({pct(by_onset(s) / len(s))}) "
        f"| {n(missed)} ({pct(missed / len(s))}) | {n(alarms)} of {n(len(others))} ({pct(alarms / len(others))}) |"
    ), by_onset(s) / len(s)


groups = [
    ("Women", patients.male == 0), ("Men", patients.male == 1),
    ("Under 50", patients.age < 50), ("50 to 64", (patients.age >= 50) & (patients.age < 65)),
    ("65 to 79", (patients.age >= 65) & (patients.age < 80)), ("80 and over", patients.age >= 80),
]
group_lines, shares = zip(*(group_row(name, mask) for name, mask in groups))
worst = groups[shares.index(min(shares))][0]
late, sirs_late = main["late_entry"], (b or a)["Modified SIRS"]["late_entry"]
time_only = (b or a)["T0"]
out["limits"] = "\n".join([
    f"{site}, {n(main['patients'])} patients. These are retrospective results on recorded data, not performance in clinical use.",
    "",
    f"- **Timely catches are rare.** Of {n(main['sepsis_patients'])} sepsis patients, {n(main['caught'])} ({pct(main['caught_share'])}) got their first alert "
    f"in the 12 hours before onset. {n(main['very_early'])} ({pct(main['very_early'] / main['sepsis_patients'])}) were alerted earlier than that. "
    f"{n(main['missed'])} ({pct(main['missed'] / main['sepsis_patients'])}) were missed.",
    f"- **False alarms.** {n(main['false_alarms'])} of {n(main['non_sepsis_patients'])} patients who never developed sepsis were alerted at least once "
    f"({pct(main['false_alarm_share'])}). That is {main['false_alarms'] / main['caught']:.1f} false alarms per timely catch. This counts patients, not alerts.",
    f"- **Workload.** {main['alerts_per_1000_patient_days']:.0f} alerts per 1,000 patient-days, and {main['repeat_alerts_per_alerted_patient']:.1f} repeat alerts per alerted patient.",
    f"- **It did not beat the clock.** A model that sees only hours in the unit scored utility {time_only['utility']:.3f}. The full model scored {main['utility']:.3f}.",
    f"- **First-day sepsis.** When the sepsis label starts before hour 24, {n(by_onset(first_day))} of {n(len(first_day))} patients were alerted by onset "
    f"({pct(by_onset(first_day) / len(first_day))}), against {n(by_onset(later))} of {n(len(later))} ({pct(by_onset(later) / len(later))}) when it starts later.",
    f"- **Late-entry cases** (label already on in the first recorded hour): the model caught {n(late['caught'])} of {n(late['sepsis_patients'])}. "
    f"The modified SIRS rule caught {n(sirs_late['caught'])} of {n(sirs_late['sepsis_patients'])}.",
    f"- **Hospital A dev numbers are calibration numbers.** The alert line was set on those {n(main_a['patients'])} patients, "
    f"so their {pct(main_a['false_alarm_share'])} false alarm share is true by construction, not a measured result.",
    "- **The score is not a probability.** It is the model's output divided by the frozen alert cutoff. Calibration was not checked.",
    "",
    f"### By age and sex, {site}",
    "",
    "| Group | Sepsis patients | Timely | Alerted by onset | Missed | False alarms |",
    "|---|---|---|---|---|---|",
    *group_lines,
    "",
    f"The group with the lowest share alerted by onset is **{worst.lower()}**. This is a first check, not a fairness audit. "
    "The data has no race, ethnicity, language or insurance fields, so those groups cannot be checked at all.",
])

# after-the-fact: how much local data a new hospital needs (scripts/v2_local_data_curve.py).
# Each arm trains on hospital A plus some hospital B patients and is scored on hospital B patients no arm trained on.
curve_file = RESULTS / "after_the_fact_v2_local_data_curve.csv"
if curve_file.exists():
    curve = pd.read_csv(curve_file)
    mixed = curve[curve.arm.str.startswith("hospital A +")].sort_values("train_patients")
    alone = curve[~curve.arm.str.startswith("hospital A +")].iloc[0]
    from_a = int(mixed.train_patients.iloc[0])                       # the arm with no local patients
    points = [{"local": int(r.train_patients) - from_a, "utility": round(float(r.utility), 3)} for r in mixed.itertuples()]
    none, most = points[0], points[-1]
    at_1000 = next(pt for pt in points if pt["local"] == 1000)
    share = (at_1000["utility"] - none["utility"]) / (most["utility"] - none["utility"])
    closes = "half" if 0.45 <= share <= 0.55 else f"{100 * share:.0f}% of"
    cap = round(100 * float(mixed.false_alarm_share.max()))
    out["local_curve"] = {
        "title": f"The first 1,000 local patients close {closes} the gap",
        "points": points,
        "alone": {"local": int(alone.train_patients), "utility": round(float(alone.utility), 3)},
        "x_title": "Hospital B patients added to training",
        "y_title": f"Utility on {n(mixed.patients.iloc[0])} hospital B test patients",
        "caption": (
            "After-the-fact: this was run after the hospital B result, so it is not part of the frozen test. "
            f"Hospital B was cut once into {n(most['local'])} local patients to learn from and {n(mixed.patients.iloc[0])} test patients "
            f"({n(mixed.sepsis_patients.iloc[0])} with sepsis) that no model trained on. Each point is a model trained on "
            f"{n(from_a)} hospital A training patients plus that many local ones. Each alert cutoff was set on the scored test patients themselves, at a {cap}% "
            "false alarm cap, the same way for every point. So this shows the shape of the curve. It is not a score to quote. "
            f"The first 1,000 local patients close {100 * share:.0f}% of the gap between none ({none['utility']:.3f}) and "
            f"{n(most['local'])} ({most['utility']:.3f}). A model trained on the {n(alone.train_patients)} local patients alone, with no "
            f"hospital A data, scored {alone.utility:.3f}. One split, so small gaps are not settled."
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
