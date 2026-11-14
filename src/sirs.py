"""The rival rule: a modified SIRS comparator.

SIRS is a bedside screening rule. It fires when two or more of these hold:

  1. heart rate over 90 beats per minute
  2. temperature over 38 or under 36 degrees C
  3. breathing rate over 20 breaths per minute, or PaCO2 under 32 mm Hg
  4. white cell count over 12 or under 4 (thousand per microliter)

It is "modified" because the full rule also counts immature white cells
(bands), and this data does not record them.

Each check uses the latest known value, carried forward, however old. A value
that was never measured counts as "does not hold."
"""


def sirs_count(features):
    """How many of the four SIRS checks hold at each hour (0 to 4)."""
    heart = features["last_HR"] > 90
    temperature = (features["last_Temp"] > 38) | (features["last_Temp"] < 36)
    breathing = (features["last_Resp"] > 20) | (features["last_PaCO2"] < 32)
    white_cells = (features["last_WBC"] > 12) | (features["last_WBC"] < 4)
    return (
        heart.astype(int) + temperature.astype(int)
        + breathing.astype(int) + white_cells.astype(int)
    )


def sirs_alert(features):
    """True at every hour where two or more SIRS checks hold."""
    return sirs_count(features) >= 2
