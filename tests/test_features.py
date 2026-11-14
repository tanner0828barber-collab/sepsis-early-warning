"""Checks that the inputs never look into the future, and that the piles are clean."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from features import LABS, VITALS, build_features, feature_groups, model_columns  # noqa: E402
from split import make_piles  # noqa: E402


def fake_patients(n_patients=12, seed=0):
    """Made-up patients with the same columns and the same gaps as the real files."""
    rng = np.random.default_rng(seed)
    tables = []
    for i in range(n_patients):
        hours = int(rng.integers(10, 40))
        t = pd.DataFrame({"pid": f"fake{i:03d}", "hosp": "A"}, index=range(hours))
        for c in VITALS:
            v = rng.normal(80, 10, hours)
            v[rng.random(hours) < 0.3] = np.nan
            t[c] = v
        for c in LABS:
            v = rng.normal(5, 1, hours)
            v[rng.random(hours) < 0.9] = np.nan
            t[c] = v
        t["Age"] = 60.0
        t["Gender"] = i % 2
        t["HospAdmTime"] = -5.0
        t["ICULOS"] = np.arange(1, hours + 1) + int(rng.integers(0, 3))
        t["SepsisLabel"] = 0
        if i % 3 == 0:
            t.loc[t.index[-6:], "SepsisLabel"] = 1
        tables.append(t)
    return pd.concat(tables, ignore_index=True)


def test_inputs_do_not_change_when_the_future_is_removed():
    full = fake_patients()
    full_features = build_features(full)
    input_columns = sum(feature_groups().values(), [])

    for keep_hours in [1, 5, 9]:
        # keep only the first few hours of every patient, then rebuild
        position = full.groupby("pid", sort=False).cumcount()
        cut = full[position < keep_hours]
        cut_features = build_features(cut)
        same_rows = full_features[position < keep_hours].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            cut_features[input_columns], same_rows[input_columns], check_dtype=False
        )


def test_inputs_do_not_change_when_the_future_is_scrambled():
    full = fake_patients(seed=1)
    before = build_features(full)
    input_columns = sum(feature_groups().values(), [])

    scrambled = full.copy()
    position = scrambled.groupby("pid", sort=False).cumcount()
    future = position >= 8
    scrambled.loc[future, VITALS + LABS] = 999.0
    after = build_features(scrambled)

    pd.testing.assert_frame_equal(
        before.loc[~future, input_columns], after.loc[~future, input_columns]
    )


def test_one_patient_never_affects_another():
    full = fake_patients(seed=2)
    all_together = build_features(full)
    one = full[full.pid == "fake005"]
    alone = build_features(one)
    input_columns = sum(feature_groups().values(), [])
    together_rows = all_together[all_together.pid == "fake005"].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        alone[input_columns], together_rows[input_columns], check_dtype=False
    )


def test_carried_forward_and_hours_since():
    t = fake_patients(n_patients=1, seed=3).iloc[:5].copy()
    t["HR"] = [np.nan, 70.0, np.nan, np.nan, 90.0]
    t["ICULOS"] = [1, 2, 3, 4, 5]
    f = build_features(t)
    assert np.isnan(f["last_HR"].iloc[0])
    assert f["last_HR"].tolist()[1:] == [70.0, 70.0, 70.0, 90.0]
    assert np.isnan(f["hours_since_HR"].iloc[0])
    assert f["hours_since_HR"].tolist()[1:] == [0.0, 1.0, 2.0, 0.0]


def test_models_see_only_their_groups():
    assert model_columns("T0") == ["hours_in_unit"]
    m1 = set(model_columns("M1"))
    assert "hours_in_unit" not in m1
    assert not any(c.startswith("hours_since_") for c in m1)
    m2 = set(model_columns("M2"))
    assert "hours_in_unit" in m2 and "admission_gap" in m2
    assert not any(c.startswith("hours_since_") for c in m2)
    m3 = set(model_columns("M3"))
    assert m2 < m3 and "labs_in_last_6h" in m3
    assert "SepsisLabel" not in m3 and "pid" not in m3
    assert not any("EtCO2" in c for c in m3)


def test_piles_do_not_share_patients():
    full = fake_patients(n_patients=60, seed=4)
    piles = make_piles(full)
    assert piles.pid.is_unique
    assert set(piles.pid) == set(full.pid)
    assert set(piles.pile) == {"fit", "tune", "dev"}


def test_rescaling_removes_a_site_wide_shift():
    """If a second site reads every heart rate 10 higher, rescaled inputs should match."""
    from rescale import rescale_features, site_norms

    site_one = fake_patients(seed=5)
    site_two = site_one.copy()
    site_two["HR"] = site_two["HR"] + 10

    one = rescale_features(build_features(site_one), site_norms(site_one))
    two = rescale_features(build_features(site_two), site_norms(site_two))
    for column in ["last_HR", "mean6_HR", "min6_HR", "max6_HR", "change6_HR"]:
        np.testing.assert_allclose(one[column], two[column], rtol=1e-9, equal_nan=True)
    # and it must not touch the measurement or time inputs
    raw = build_features(site_one)
    pd.testing.assert_series_equal(one["hours_since_HR"], raw["hours_since_HR"])
    pd.testing.assert_series_equal(one["hours_in_unit"], raw["hours_in_unit"])
