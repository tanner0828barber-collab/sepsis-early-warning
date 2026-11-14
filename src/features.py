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
    }


# The four planned models, by which groups they are allowed to see.
# T0 is the odd one out: it sees a single column, hours in the unit.
MODEL_GROUPS = {
    "M1": ["physiology", "trends", "demographics"],
    "M2": ["physiology", "trends", "demographics", "time"],
    "M3": ["physiology", "trends", "demographics", "time", "measurement"],
}


def model_columns(name):
    """Return the input columns for T0, M1, M2 or M3."""
    if name == "T0":
        return ["hours_in_unit"]
    groups = feature_groups()
    columns = []
    for group in MODEL_GROUPS[name]:
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

    keys = table[["pid", "hosp", "ICULOS", "SepsisLabel"]]
    return pd.concat([keys, pd.DataFrame(out)], axis=1)
