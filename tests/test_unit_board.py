"""Checks for the app's Unit board (app/unit_board.py).

The board must not show the future, must rank by the score, and must keep
what happened hidden until its hour gets there.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import unit_board  # noqa: E402
from features import build_features  # noqa: E402
from train import score_rows  # noqa: E402


def test_the_board_has_the_patients_the_rule_asked_for():
    assert len(unit_board.PATIENTS) == 12
    assert sum(p["sepsis"] for p in unit_board.PATIENTS.values()) == unit_board.UNIT["with_sepsis"] == 3
    assert sorted(unit_board.BEDS.values()) == list(range(1, 13))


def test_a_score_on_the_board_uses_only_the_hours_before_it():
    # Cut each record off at an hour and score it again. The board must show the same number.
    for pid, p in unit_board.PATIENTS.items():
        rows = unit_board.RECORDS[unit_board.RECORDS.pid == pid].reset_index(drop=True).assign(hosp="saved")
        for hour in {p["first"], (p["first"] + p["last"]) // 2, p["last"]}:
            cut = build_features(rows[rows.ICULOS <= hour])
            again = score_rows(unit_board.MODEL, cut, unit_board.COLUMNS, unit_board.FROZEN.get("smooth", 1))[-1] / unit_board.THRESHOLD
            assert unit_board.at_hour(pid, hour)["relative"] == pytest.approx(again)


def test_patients_are_ranked_highest_score_first():
    for hour in [unit_board.FIRST_HOUR, 12, 30, 48, unit_board.LAST_HOUR]:
        here, away = unit_board.ranked(hour)
        scores = [bed["relative"] for bed in here]
        assert scores == sorted(scores, reverse=True)
        assert len(here) + len(away) == 12
        assert all(bed["first"] <= hour <= bed["last"] for bed in here)


def test_what_happened_stays_hidden_until_the_board_gets_there():
    for pid, p in unit_board.PATIENTS.items():
        reveal = min(p["onset"], p["last"]) if p["sepsis"] else p["last"]
        if reveal > p["first"]:
            assert unit_board.at_hour(pid, reveal - 1)["happened"] is None
        assert unit_board.at_hour(pid, reveal)["happened"] is not None


def test_the_three_words_for_the_score():
    assert unit_board.level(1.0)[1] == "Over the line"
    assert unit_board.level(0.99)[1] == "Near the line"
    assert unit_board.level(unit_board.NEAR)[1] == "Near the line"
    assert unit_board.level(unit_board.NEAR - 0.01)[1] == "Below the line"


def test_alert_words_match_the_score():
    for hour in range(unit_board.FIRST_HOUR, 61):
        for bed in unit_board.ranked(hour)[0]:
            assert (bed["status"] == "over") == (bed["relative"] >= 1)
            if bed["status"] == "earlier":
                assert bed["first_alert"] < hour and bed["relative"] < 1
            if bed["status"] == "quiet":
                assert bed["first_alert"] is None
