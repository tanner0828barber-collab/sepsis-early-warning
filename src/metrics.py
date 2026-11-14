"""Turn hourly scores into per-patient outcomes, using DEFINITIONS.md.

Everything here is counted per patient, not per row:

  caught       sepsis patient, first alert between 12 hours before onset and onset
  very early   sepsis patient, first alert more than 12 hours before onset
  missed       sepsis patient, no alert by onset
  false alarm  patient who never develops sepsis, alerted at least once
  quiet        patient who never develops sepsis, never alerted

Onset is label start plus 6 hours. The hour clock is ICULOS, hours in the unit.
"""
import numpy as np
import pandas as pd

ONSET_AFTER_LABEL = 6   # the challenge switched the label on 6 hours before onset
REWARD_WINDOW = 12      # the challenge's reward window opens 12 hours before onset


def threshold_for_budget(rows, scores, budget):
    """Find the score cutoff that alerts on at most `budget` of non-sepsis patients.

    `rows` needs pid and SepsisLabel. A patient is alerted if any hour's score
    is at or above the cutoff, so we look at each patient's highest score.
    """
    frame = pd.DataFrame({
        "pid": rows["pid"].to_numpy(),
        "label": rows["SepsisLabel"].to_numpy(),
        "score": np.asarray(scores),
    })
    per_patient = frame.groupby("pid", sort=False).agg(
        sepsis=("label", "max"), top=("score", "max")
    )
    top_scores = np.sort(per_patient.loc[per_patient.sepsis == 0, "top"].to_numpy())[::-1]
    allowed = int(np.floor(budget * len(top_scores)))
    if allowed < 1:
        return float("inf")
    # the cutoff sits halfway between the last patient we may alert and the next one
    return float((top_scores[allowed - 1] + top_scores[allowed]) / 2)


def patient_outcomes(rows, alert_on):
    """One row per patient: did we alert, when, and what that counts as.

    `rows` needs pid, ICULOS and SepsisLabel, whole records in time order.
    `alert_on` is True for every hour the score is at or above the threshold
    (or, for a rule like SIRS, every hour the rule fires).
    """
    frame = pd.DataFrame({
        "pid": rows["pid"].to_numpy(),
        "hour": rows["ICULOS"].to_numpy(),
        "label": rows["SepsisLabel"].to_numpy(),
        "on": np.asarray(alert_on, dtype=bool),
    })
    grouped = frame.groupby("pid", sort=False)

    previous_on = grouped["on"].shift(1, fill_value=False)
    frame["crossing"] = frame["on"] & ~previous_on          # a separate alert starts here
    frame["alert_hour"] = frame["hour"].where(frame["on"])
    frame["label_hour"] = frame["hour"].where(frame["label"] == 1)
    grouped = frame.groupby("pid", sort=False)

    patients = grouped.agg(
        sepsis=("label", "max"),
        first_label=("label", "first"),
        first_hour=("hour", "first"),
        last_hour=("hour", "last"),
        hours=("hour", "size"),
        label_start=("label_hour", "min"),
        first_alert=("alert_hour", "min"),
        alerts=("crossing", "sum"),
    )
    patients["sepsis"] = patients["sepsis"] == 1
    patients["late_entry"] = patients["first_label"] == 1
    patients["onset"] = patients["label_start"] + ONSET_AFTER_LABEL
    patients["hours_of_warning"] = patients["onset"] - patients["first_alert"]

    alerted = patients["first_alert"].notna()
    warning = patients["hours_of_warning"]
    outcome = pd.Series("quiet", index=patients.index)
    outcome[~patients.sepsis & alerted] = "false alarm"
    outcome[patients.sepsis] = "missed"
    outcome[patients.sepsis & alerted & (warning >= 0) & (warning <= REWARD_WINDOW)] = "caught"
    outcome[patients.sepsis & alerted & (warning > REWARD_WINDOW)] = "very early"
    patients["outcome"] = outcome
    # a sepsis patient alerted only after onset is still a miss; we keep count of them
    patients["alert_after_onset"] = patients.sepsis & alerted & (warning < 0)
    return patients.drop(columns=["first_label"])


def summarize(patients):
    """Count the outcomes for a group of patients."""
    sepsis = patients[patients.sepsis]
    others = patients[~patients.sepsis]
    counts = patients["outcome"].value_counts()
    caught = int(counts.get("caught", 0))
    very_early = int(counts.get("very early", 0))
    missed = int(counts.get("missed", 0))
    false_alarms = int(counts.get("false alarm", 0))
    alerted = patients[patients.alerts > 0]
    in_time = sepsis[sepsis.outcome.isin(["caught", "very early"])]
    patient_days = patients["hours"].sum() / 24

    def share(part, whole):
        return float(part / whole) if whole else float("nan")

    return {
        "patients": int(len(patients)),
        "sepsis_patients": int(len(sepsis)),
        "caught": caught,
        "very_early": very_early,
        "missed": missed,
        "missed_but_alerted_after_onset": int(sepsis.alert_after_onset.sum()),
        "alerted_by_onset_share": share(caught + very_early, len(sepsis)),
        "caught_share": share(caught, len(sepsis)),
        "non_sepsis_patients": int(len(others)),
        "false_alarms": false_alarms,
        "false_alarm_share": share(false_alarms, len(others)),
        "false_alerts_per_true_alert": share(false_alarms, caught + very_early),
        "median_hours_of_warning": float(in_time.hours_of_warning.median()) if len(in_time) else float("nan"),
        "alerts_per_1000_patient_days": share(1000 * patients.alerts.sum(), patient_days),
        "repeat_alerts_per_alerted_patient": share((alerted.alerts - 1).sum(), len(alerted)),
    }


def alerted_by_onset_share(rows, scores, threshold):
    """Share of sepsis patients with an alert at or before onset. Used for tuning."""
    patients = patient_outcomes(rows, np.asarray(scores) >= threshold)
    return summarize(patients)["alerted_by_onset_share"]
