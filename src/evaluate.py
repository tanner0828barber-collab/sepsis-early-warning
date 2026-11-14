"""One function that produces every reported number for a set of alerts.

Used the same way for the model and for the SIRS rule, and the same way on
hospital A dev and on hospital B, so the columns of the scoreboard are
always computed by identical code.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import patient_outcomes, summarize
from utility import normalize, utility_parts


def evaluate_alerts(rows, alert_on, scores=None):
    """Return (numbers, per-patient table) for one set of hourly alerts.

    `alert_on` is True at every hour the model or rule is alerting.
    `scores` is optional: with it we also report hourly AUROC and AUPRC.
    """
    alert_on = np.asarray(alert_on, dtype=bool)
    patients = patient_outcomes(rows, alert_on)
    numbers = summarize(patients)
    # the official utility score, kept per patient so ranges can resample patients
    observed, best, inaction = utility_parts(rows, alert_on)
    patients["u_observed"], patients["u_best"], patients["u_inaction"] = observed, best, inaction
    numbers["utility"] = normalize(observed, best, inaction)

    if scores is not None:
        labels = rows["SepsisLabel"].to_numpy()
        numbers["hourly_auroc"] = float(roc_auc_score(labels, scores))
        numbers["hourly_auprc"] = float(average_precision_score(labels, scores))

    # late-entry cases: the label is already on in the patient's first row
    for name, group in [
        ("late_entry", patients[patients.sepsis & patients.late_entry]),
        ("with_history", patients[patients.sepsis & ~patients.late_entry]),
    ]:
        counts = group["outcome"].value_counts()
        numbers[name] = {
            "sepsis_patients": int(len(group)),
            "caught": int(counts.get("caught", 0)),
            "very_early": int(counts.get("very early", 0)),
            "missed": int(counts.get("missed", 0)),
        }
    return numbers, patients
