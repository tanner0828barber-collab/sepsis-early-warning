"""A label-free adjustment for a new hospital: rescale values by that site's own norms.

The worry: the same vital sign or lab may run a little higher or lower at
another hospital, because of different devices, different patients, or
different recording habits. A model that learned "heart rate above 95" at
hospital A would then be asking the wrong question at hospital B.

The attempt: express every value as "how far from this hospital's own
average, in units of this hospital's own spread." That needs each hospital's
measurements but not its sepsis labels.

This is one of the planned travel attempts. It is reported as a separate,
labeled row, never as the main result.
"""
import pandas as pd

from features import LABS, VITALS


def site_norms(raw_table):
    """Average and spread of every measured vital sign and lab at one hospital."""
    values = raw_table[VITALS + LABS]
    return pd.DataFrame({"average": values.mean(), "spread": values.std()})


def rescale_features(features, norms):
    """Return a copy of the inputs with values expressed relative to the site's norms."""
    out = features.copy()
    for c in VITALS + LABS:
        average, spread = norms.loc[c, "average"], norms.loc[c, "spread"]
        if not spread > 0:
            continue  # never measured, or a single repeated value: leave as is
        out[f"last_{c}"] = (out[f"last_{c}"] - average) / spread
        if c in VITALS:
            for kind in ["mean6", "min6", "max6"]:
                out[f"{kind}_{c}"] = (out[f"{kind}_{c}"] - average) / spread
            out[f"change6_{c}"] = out[f"change6_{c}"] / spread  # a change has no average to remove
    return out
