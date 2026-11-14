"""Turn the raw hourly table into model inputs.

The one rule: every input for a patient at hour t uses only that patient's
rows at hour t or earlier. Nothing from the future, nothing from other
patients. tests/test_features.py checks this by cutting records short and
confirming the inputs for the earlier hours do not change.

Inputs come in named groups so we can train models that see only some of them:

  physiology    latest known value of each vital sign and lab (carried forward)
  trends        how each vital sign moved over the last 6 hours
  demographics  age and sex
  time          hours in the unit, and the gap between hospital and unit admission
  measurement   how long ago each value was measured, and how many labs appeared

A lab value appearing in a window is a sign that something was measured. It is
not proof a test was ordered at that moment or that a clinician suspected sepsis.
"""
import pandas as pd

from sirs import sirs_count

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
LABS = [
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets",
]
# EtCO2 is left out: it is empty in 100.0% of hospital A rows, so a model
# trained there can learn nothing from it.
DROPPED = ["EtCO2"]

WINDOW = 6  # hours


def feature_groups():
    """Return the input column names in each group."""
    values = VITALS + LABS
    return {
        "physiology": [f"last_{c}" for c in values],
        "trends": (
            [f"change6_{c}" for c in VITALS]
            + [f"mean6_{c}" for c in VITALS]
            + [f"min6_{c}" for c in VITALS]
            + [f"max6_{c}" for c in VITALS]
        ),
        "demographics": ["Age", "Gender"],
        "time": ["hours_in_unit", "admission_gap"],
        "measurement": (
            [f"hours_since_{c}" for c in values]
            + ["labs_in_last_6h", "labs_so_far"]
        ),
        # The three groups below were added in the improvement round.
        # They follow what the top challenge teams reported using.
        "scores": [
            "shock_index", "pulse_pressure", "bun_creatinine_ratio",
            "sofa_partial", "qsofa_partial", "sirs_count",
        ],
        "long_trends": (
            [f"spread6_{c}" for c in VITALS]
            + [f"mean24_{c}" for c in VITALS]
            + [f"min24_{c}" for c in VITALS]
            + [f"max24_{c}" for c in VITALS]
            + [f"delta_{c}" for c in LABS]
        ),
        "measurement_extra": [
            "distinct_labs_so_far", "labs_per_hour_so_far",
            "hours_since_any_lab", "vitals_empty_last_6h",
        ],
    }


# The planned models, by which groups they are allowed to see.
# T0 is the odd one out: it sees a single column, hours in the unit.
MODEL_GROUPS = {
    "M1": ["physiology", "trends", "demographics"],
    "M2": ["physiology", "trends", "demographics", "time"],
    "M3": ["physiology", "trends", "demographics", "time", "measurement"],
}
# Where each added group belongs: body signals join at M1, measurement signals at M3.
EXTRA_GROUPS = {
    "M1": ["scores", "long_trends"],
    "M2": ["scores", "long_trends"],
    "M3": ["scores", "long_trends", "measurement_extra"],
}
# Set by the cross-validation result in the improvement round (see DECISIONS.md).
USE_EXTRA_INPUTS = False


def model_columns(name, extra=None):
    """Return the input columns for T0, M1, M2 or M3.

    `extra` says whether to include the inputs added in the improvement round.
    Left as None, it follows USE_EXTRA_INPUTS.
    """
    if name == "T0":
        return ["hours_in_unit"]
    if extra is None:
        extra = USE_EXTRA_INPUTS
    groups = feature_groups()
    wanted = MODEL_GROUPS[name] + (EXTRA_GROUPS[name] if extra else [])
    columns = []
    for group in wanted:
        columns += groups[group]
    return columns


def build_features(table):
    """Build every input column for every hourly row.

    `table` must hold whole patient records in time order, as load_hospital
    returns them. The result has one row per input row, in the same order.
    """
    table = table.reset_index(drop=True)
    pid = table["pid"]
    by_patient = table.groupby("pid", sort=False)
    values = VITALS + LABS

    out = {}  # column name -> values, joined into one table at the end

    # physiology: the latest known value, carried forward and never backward
    last = by_patient[values].ffill()
    for c in values:
        out[f"last_{c}"] = last[c]

    # trends, vital signs only
    # change6: latest known value now, minus the latest known value 6 hours ago
    earlier = last[VITALS].groupby(pid, sort=False).shift(WINDOW)
    for c in VITALS:
        out[f"change6_{c}"] = last[c] - earlier[c]
    # mean, lowest and highest of the values actually measured in the last 6 hours
    rolling = by_patient[VITALS].rolling(WINDOW, min_periods=1)
    for name, result in [
        ("mean6", rolling.mean()),
        ("min6", rolling.min()),
        ("max6", rolling.max()),
    ]:
        result = result.reset_index(level=0, drop=True).sort_index()
        for c in VITALS:
            out[f"{name}_{c}"] = result[c]

    # demographics
    out["Age"] = table["Age"]
    out["Gender"] = table["Gender"]

    # time
    out["hours_in_unit"] = table["ICULOS"]
    out["admission_gap"] = table["HospAdmTime"]

    # measurement: hours since each value was last measured (empty if never)
    hour = table["ICULOS"]
    measured = table[values].notna()
    hour_if_measured = measured.mul(hour, axis=0).where(measured)
    last_measured_hour = hour_if_measured.groupby(pid, sort=False).ffill()
    for c in values:
        out[f"hours_since_{c}"] = hour - last_measured_hour[c]

    # measurement: how many lab values appeared
    labs_this_hour = measured[LABS].sum(axis=1)
    labs_grouped = labs_this_hour.groupby(pid, sort=False)
    out["labs_in_last_6h"] = (
        labs_grouped.rolling(WINDOW, min_periods=1).sum()
        .reset_index(level=0, drop=True).sort_index()
    )
    out["labs_so_far"] = labs_grouped.cumsum()

    # ---- inputs added in the improvement round ----

    # scores: bedside ratios and partial clinical scores, from the latest known values.
    # A part whose input was never measured counts as 0, so these are "at least" scores.
    out["shock_index"] = last["HR"] / last["SBP"]
    out["pulse_pressure"] = last["SBP"] - last["DBP"]
    out["bun_creatinine_ratio"] = last["BUN"] / last["Creatinine"]
    platelets, bilirubin, creatinine = last["Platelets"], last["Bilirubin_total"], last["Creatinine"]
    out["sofa_partial"] = (
        (platelets < 150).astype(int) + (platelets < 100) + (platelets < 50) + (platelets < 20)
        + (bilirubin >= 1.2).astype(int) + (bilirubin >= 2.0) + (bilirubin >= 6.0) + (bilirubin >= 12.0)
        + (creatinine >= 1.2).astype(int) + (creatinine >= 2.0) + (creatinine >= 3.5) + (creatinine >= 5.0)
        + (last["MAP"] < 70).astype(int)
    )
    out["qsofa_partial"] = (last["SBP"] <= 100).astype(int) + (last["Resp"] >= 22)
    out["sirs_count"] = sirs_count(last.add_prefix("last_"))

    # long trends: how much each vital sign bounced around, and a 24 hour view
    spread = rolling.std().reset_index(level=0, drop=True).sort_index()
    for c in VITALS:
        out[f"spread6_{c}"] = spread[c]
    day = by_patient[VITALS].rolling(24, min_periods=1)
    for name, result in [("mean24", day.mean()), ("min24", day.min()), ("max24", day.max())]:
        result = result.reset_index(level=0, drop=True).sort_index()
        for c in VITALS:
            out[f"{name}_{c}"] = result[c]
    # delta: for each lab, the latest measurement minus the one before it
    deltas = pd.DataFrame(index=table.index, columns=LABS, dtype=float)
    for c in LABS:
        was_measured = measured[c]
        deltas.loc[was_measured, c] = table.loc[was_measured, c].groupby(pid[was_measured], sort=False).diff()
    deltas = deltas.groupby(pid, sort=False).ffill()
    for c in LABS:
        out[f"delta_{c}"] = deltas[c]

    # more measurement signals
    out["distinct_labs_so_far"] = last[LABS].notna().sum(axis=1)
    hours_seen = by_patient.cumcount() + 1
    out["labs_per_hour_so_far"] = out["labs_so_far"] / hours_seen
    out["hours_since_any_lab"] = (-last_measured_hour[LABS]).add(hour, axis=0).min(axis=1)
    empty_vitals = (~measured[VITALS]).sum(axis=1)
    out["vitals_empty_last_6h"] = (
        empty_vitals.groupby(pid, sort=False).rolling(WINDOW, min_periods=1).sum()
        .reset_index(level=0, drop=True).sort_index()
    )

    keys = table[["pid", "hosp", "ICULOS", "SepsisLabel"]]
    return pd.concat([keys, pd.DataFrame(out)], axis=1)
