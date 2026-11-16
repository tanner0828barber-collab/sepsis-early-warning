"""The Unit board: 12 saved hospital B patients at one hour, ranked by the model's score.

The user is a nurse deciding which patient to look at first. The Replay tab
shows one patient. This shows a whole made-up unit at once, highest score at
the top, so the question "who first?" has an answer on screen.

It uses the same frozen model and the same frozen alert line as the Replay.
The patients were picked by scripts/build_unit_board.py with a seeded draw
that never reads the model's scores. What happened to each patient stays
hidden until the board's hour reaches it.

This file draws HTML only and does not import Gradio, so the checks in
tests/test_unit_board.py can run it directly.

This project is a prototype and is not medical advice.
"""
import json
import math
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features  # noqa: E402
from train import score_rows  # noqa: E402

APP_DATA = Path(__file__).resolve().parent / "data"

FROZEN = json.loads((ROOT / "results" / "frozen.json").read_text())
THRESHOLD = FROZEN["models"]["M3"]["threshold"]
COLUMNS = FROZEN["models"]["M3"]["columns"]
MODEL = joblib.load(ROOT / "models" / "M3.joblib")

UNIT = json.loads((APP_DATA / "unit.json").read_text())
RECORDS = pd.read_csv(APP_DATA / "unit_patients.csv")
BEDS = {bed["pid"]: bed["bed"] for bed in UNIT["beds"]}

NEAR = 0.5            # "Near the line" is half the alert line or more. A display choice, not a clinical standard.
STALE_AFTER = 4       # hours; a vital sign older than this is marked. The same display choice as the Replay.
REWARD_WINDOW = 12    # a first alert in the 12 hours before onset is "timely", as everywhere else in the project
START_HOUR = 12       # the board opens half a day in, when every bed has measurements to show
VITALS = [("HR", "Heart rate", "HR"), ("Temp", "Temperature", "Temp"), ("Resp", "Breathing rate", "Resp"), ("SBP", "Systolic blood pressure", "SBP")]


def level(relative):
    """The score against the alert line, as a word a nurse can read at a glance."""
    if relative >= 1:
        return "over", "Over the line"
    if relative >= NEAR:
        return "near", "Near the line"
    return "below", "Below the line"


def _load():
    """Score every saved record once, hour by hour.

    Each input at hour t uses only that patient's rows at hour t or earlier
    (tests/test_features.py checks this), so the score at hour t is what the
    model would have said at hour t.
    """
    patients = {}
    for pid, rows in RECORDS.groupby("pid", sort=False):
        rows = rows.reset_index(drop=True).assign(hosp="saved")
        seen = build_features(rows)
        seen["relative"] = score_rows(MODEL, seen, COLUMNS, FROZEN.get("smooth", 1)) / THRESHOLD
        labeled = rows[rows.SepsisLabel == 1]
        label_start = int(labeled.ICULOS.iloc[0]) if len(labeled) else None
        patients[pid] = {
            "pid": pid, "bed": BEDS[pid], "seen": seen.set_index("ICULOS"),
            "first": int(rows.ICULOS.iloc[0]), "last": int(rows.ICULOS.iloc[-1]),
            "age": int(rows.Age.iloc[0]), "who": "Man" if rows.Gender.iloc[0] == 1 else "Woman",
            "sepsis": label_start is not None,
            "onset": None if label_start is None else label_start + 6,    # the label starts 6 hours before onset
            "late_entry": bool(rows.SepsisLabel.iloc[0] == 1),
        }
    return patients


PATIENTS = _load()
FIRST_HOUR = min(p["first"] for p in PATIENTS.values())
LAST_HOUR = max(p["last"] for p in PATIENTS.values())


def plural(n, word):
    n = int(n)
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def at_hour(pid, hour):
    """One bed at one hour: what the nurse would see, and what may be revealed yet."""
    p = PATIENTS[pid]
    here = {"pid": pid, "bed": p["bed"], "age": p["age"], "who": p["who"], "first": p["first"], "last": p["last"]}
    if hour < p["first"]:
        return {**here, "place": "not yet", "happened": None}

    upto = min(hour, p["last"])
    seen = p["seen"].loc[:upto]
    alerting = seen.index[seen["relative"] >= 1]
    first_alert = int(alerting[0]) if len(alerting) else None
    now = seen.iloc[-1]
    relative = float(now["relative"])
    status = "over" if relative >= 1 else "earlier" if first_alert is not None else "quiet"

    # What happened, shown only once the board gets there: at the onset hour for a sepsis patient
    # (or the end of the record if it stops first), at the end of the record for everyone else.
    happened = None
    if p["sepsis"] and hour >= min(p["onset"], p["last"]):
        when = f"at or before hour {p['onset']}" if p["late_entry"] else f"at hour {p['onset']}"
        if p["onset"] > p["last"]:
            when += ", after the record ends"
        in_time = first_alert is not None and first_alert <= p["onset"]
        warned = p["onset"] - first_alert if in_time else None
        if not in_time:
            kind, text = "missed", f"Sepsis onset {when}. No alert in time. Missed."
            if first_alert is not None:
                text += f" The first alert came after onset, at hour {first_alert}."
        elif p["late_entry"]:
            kind, text = "caught", f"Sepsis onset {when}. First alert at hour {first_alert}."
        elif warned <= REWARD_WINDOW:
            kind, text = "caught", f"Sepsis onset {when}. First alert {plural(warned, 'hour')} before. Timely."
        else:
            kind, text = "early", f"Sepsis onset {when}. First alert {plural(warned, 'hour')} before. Very early."
        happened = {"kind": kind, "text": text}
    elif not p["sepsis"] and hour >= p["last"]:
        if first_alert is None:
            happened = {"kind": "fine", "text": "Never developed sepsis. No alert."}
        else:
            happened = {"kind": "false", "text": f"Never developed sepsis. The alert at hour {first_alert} was a false alarm."}

    vitals = []
    for column, name, short in VITALS:
        value, age = now[f"last_{column}"], now[f"hours_since_{column}"]
        vitals.append({"name": name, "short": short, "value": None if pd.isna(value) else float(value), "age": None if pd.isna(age) else int(age)})
    return {
        **here, "place": "ended" if hour > p["last"] else "here", "relative": relative, "status": status,
        "first_alert": first_alert, "vitals": vitals, "happened": happened,
    }


def ranked(hour):
    """Every bed at this hour. Patients in the unit come first, highest score at the top."""
    beds = [at_hour(pid, hour) for pid in PATIENTS]
    here = sorted((b for b in beds if b["place"] == "here"), key=lambda b: (-b["relative"], b["bed"]))
    away = sorted((b for b in beds if b["place"] != "here"), key=lambda b: b["bed"])
    return here, away


# ---------------------------------------------------------------- drawing
SHAPES = {
    "over": '<path d="M8 1 15.3 14.5H.7Z" fill="currentColor"/>',
    "earlier": '<path d="M8 3.6 13.2 13.2H2.8Z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>',
    "quiet": '<circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="2.6"/>',
}
STATUS_WORDS = {"over": "Alerting now", "earlier": "Alerted earlier", "quiet": "Quiet"}
LOW, HIGH = math.log10(0.03), math.log10(30)   # the same log scale as the Replay's score chart


def _across(relative):
    """How far along the meter a score sits, 0 to 100."""
    return 100 * (math.log10(min(max(relative, 0.03), 30)) - LOW) / (HIGH - LOW)


def _meter(relative):
    kind, word = level(relative)
    return (
        f'<div class="ub-level {kind}"><div class="ub-meter" role="img" aria-label="{word}: {relative:.2f} times the alert line">'
        f'<i style="width:{_across(relative):.1f}%"></i><u style="left:{_across(1):.1f}%"></u></div>'
        f'<span><b>{word}</b> {relative:.2f} times the alert line</span></div>'
    )


def _vitals(vitals):
    cells = []
    for v in vitals:
        if v["value"] is None:
            cells.append(f'<span title="{v["name"]}"><small>{v["short"]}</small> <em>not measured</em></span>')
            continue
        old = f' <em>{plural(v["age"], "hour")} old</em>' if v["age"] > STALE_AFTER else ""
        cells.append(f'<span title="{v["name"]}"><small>{v["short"]}</small> <b>{v["value"]:g}</b>{old}</span>')
    return f'<div class="ub-vitals">{"".join(cells)}</div>'


def _happened(happened):
    if happened is None:
        # a quiet bar, not the same three words on every row; the words stay for screen readers and on hover
        return '<div class="ub-happened shut" title="Not revealed yet"><i aria-hidden="true"></i><span>Not revealed yet</span></div>'
    return f'<div class="ub-happened {happened["kind"]}">{happened["text"]}</div>'


def _who(bed):
    age = "90 or older" if bed["age"] >= 90 else str(bed["age"])
    return f'<div class="ub-bed"><b class="display">Bed {bed["bed"]}</b><span>{bed["who"]}, {age}</span></div>'


def _row(rank, bed):
    status = bed["status"]
    words = STATUS_WORDS[status]
    if status == "earlier":
        words += f'<small>at hour {bed["first_alert"]}, under the line now</small>'
    elif status == "over":
        words += f'<small>first at hour {bed["first_alert"]}</small>'
    return (
        f'<li class="ub-row {status}"><div class="ub-rank display" aria-label="Number {rank} to look at">{rank}</div>{_who(bed)}'
        f'<div class="ub-status"><svg viewBox="0 0 16 16" aria-hidden="true">{SHAPES[status]}</svg><span>{words}</span></div>'
        f'{_meter(bed["relative"])}{_vitals(bed["vitals"])}{_happened(bed["happened"])}</li>'
    )


def _away_row(bed):
    if bed["place"] == "not yet":
        note = f'<div class="ub-status"><span>Record starts at hour {bed["first"]}</span></div>'
        return f'<li class="ub-row away"><div class="ub-rank display" aria-hidden="true"></div>{_who(bed)}{note}{_happened(None)}</li>'
    note = f'<div class="ub-status"><span>Record ended at hour {bed["last"]}</span></div>'
    return f'<li class="ub-row away"><div class="ub-rank display" aria-hidden="true"></div>{_who(bed)}{note}{_happened(bed["happened"])}</li>'


def board(hour):
    """The whole Unit board at one hour, as HTML."""
    hour = int(min(max(round(hour), FIRST_HOUR), LAST_HOUR))
    here, away = ranked(hour)
    over = sum(1 for bed in here if bed["status"] == "over")
    lead = (
        f"{plural(len(here), 'patient')} in the unit at this hour, ranked by the model's score, highest first. "
        f"{'None is' if over == 0 else plural(over, 'patient') + (' is' if over == 1 else ' are')} over the alert line now. "
        "What happened to each patient stays hidden until the board reaches it."
    )
    rows = "".join(_row(rank, bed) for rank, bed in enumerate(here, start=1))
    listing = f'<ol class="ub-list">{rows}</ol>' if here else '<p class="finding">Every record has ended by this hour.</p>'
    gone = ""
    if away:
        gone = (
            '<section class="tile wide12"><h3>Not in the unit at this hour</h3>'
            f'<ol class="ub-list">{"".join(_away_row(bed) for bed in away)}</ol></section>'
        )
    share = round(UNIT["hospital_patients"] / UNIT["hospital_sepsis_patients"])
    return (
        '<div class="grid12">'
        f'<section class="tile wide12"><h3>Unit board, hour {hour} in the unit</h3>'
        f'<div class="headline display">Who to look at first</div><p class="finding">{lead}</p></section>'
        '<section class="tile wide12"><div class="ub-head" aria-hidden="true"><span></span><span>Bed</span><span>Alert</span>'
        f'<span>How close to an alert</span><span>Latest vital signs</span><span>What happened</span></div>{listing}</section>{gone}</div>'
        f'<p class="source">A made-up unit: {len(PATIENTS)} real, public, de-identified {UNIT["site"][0].lower() + UNIT["site"][1:]} records, '
        f'lined up by hour in the unit. {UNIT["with_sepsis"]} of the {len(PATIENTS)} develop sepsis, against about 1 in {share} at the hospital, '
        "on purpose. Retrospective replay of saved records, not a live monitor. This project is a prototype and is not medical advice.</p>"
        '<details class="more"><summary>How this board was built</summary>'
        f'<p class="source">These are {len(PATIENTS)} real, public, de-identified records from the PhysioNet / Computing in '
        f'Cardiology Challenge 2019, {UNIT["site"][0].lower() + UNIT["site"][1:]}, picked by rule and not by eye: {UNIT["rule"]}. '
        "They were not in the unit together. The board lines their records up by hour in the unit, and the bed numbers are made up. "
        f'{UNIT["with_sepsis"]} of the {len(PATIENTS)} develop sepsis. At this hospital it was {UNIT["hospital_sepsis_patients"]:,} of '
        f'{UNIT["hospital_patients"]:,}, about 1 in {share}, so sepsis is over-represented here on purpose. '
        f'"Near the line" means {NEAR:g} of the alert line or more. That level is my own choice for this display, and so is marking a vital sign '
        f"older than {STALE_AFTER} hours. HR is heart rate in beats per minute, Temp is temperature in °C, Resp is breathing rate in breaths per "
        "minute, and SBP is systolic blood pressure in mm Hg. What happened to each patient shows only when the board reaches it. The model "
        "never sees it. The score is the model's output divided by the frozen alert cutoff, not a probability.</p></details>"
    )


def css(c):
    """The board's styles. `c` holds the app's own colors, so the two screens match."""
    return f"""
.ub-head, .ub-row {{ display: grid; grid-template-columns: 44px 128px 168px minmax(180px, 1.1fr) minmax(190px, 1.2fr) minmax(170px, 1.2fr); gap: 6px 16px; align-items: center; }}
.ub-head {{ color: {c['FOG']}; font-size: 14px; font-weight: 600; padding: 0 12px 8px; }}
.gradio-container ol.ub-list {{ list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 8px; }}
.gradio-container .ub-row {{ background: {c['LANE']}; border: 1px solid transparent; border-radius: 16px; padding: 12px; margin: 0; color: {c['CHALK']}; }}
.ub-row.over {{ border-color: rgba(255,107,61,.55); background: linear-gradient(90deg, rgba(255,107,61,.16), rgba(255,107,61,.03) 60%), {c['LANE']}; }}
.ub-row.away {{ background: transparent; border: 1px dashed {c['RULE']}; }}
.ub-row.away .ub-status {{ grid-column: 3 / 6; font-weight: 400; color: {c['FOG']}; }} .ub-row.away .ub-bed b {{ color: {c['FOG']}; }}
.ub-rank {{ font-weight: 800; font-size: 30px; line-height: 1; color: {c['FOG']}; text-align: center; }}
.ub-row.over .ub-rank {{ color: {c['FLARE']}; }}
.ub-bed b {{ display: block; font-weight: 700; font-size: 19px; line-height: 1.2; }}
.ub-bed span {{ color: {c['FOG']}; font-size: 14px; }}
.ub-status {{ display: flex; align-items: center; gap: 8px; font-weight: 600; line-height: 1.25; }}
.ub-status svg {{ width: 16px; height: 16px; flex: none; color: {c['FOG']}; }}
.ub-row.over .ub-status, .ub-row.over .ub-status svg, .ub-row.earlier .ub-status svg {{ color: {c['FLARE']}; }}
.ub-status small {{ display: block; font-weight: 400; font-size: 13px; color: {c['FOG']}; }}
.ub-meter {{ position: relative; height: 10px; border-radius: 999px; background: {c['NIGHT']}; overflow: visible; margin: 2px 0 6px; }}
.ub-meter i {{ display: block; height: 100%; border-radius: 999px; background: #56616F; }}
.ub-level.near .ub-meter i {{ background: {c['AMBER']}; }} .ub-level.over .ub-meter i {{ background: {c['FLARE']}; }}
.ub-meter u {{ position: absolute; top: -4px; bottom: -4px; width: 2px; background: {c['CHALK']}; }}
.ub-level span {{ color: {c['FOG']}; font-size: 13.5px; }} .ub-level span b {{ color: {c['CHALK']}; font-weight: 600; font-size: 15px; margin-right: 4px; }}
.ub-level.near span b {{ color: {c['AMBER']}; }} .ub-level.over span b {{ color: {c['FLARE']}; }}
.ub-vitals {{ display: grid; grid-template-columns: 1fr 1fr; gap: 2px 12px; font-variant-numeric: tabular-nums; }}
.ub-vitals small {{ color: {c['FOG']}; font-size: 13px; }} .ub-vitals b {{ font-weight: 600; }}
.ub-vitals em {{ font-style: normal; color: {c['AMBER']}; font-size: 13px; white-space: nowrap; }} .ub-vitals b + em {{ display: block; }}
.ub-happened {{ font-size: 14.5px; line-height: 1.4; }}
.ub-happened.shut {{ color: {c['FOG']}; }}
.ub-happened.shut i {{ display: inline-block; width: 22px; height: 2px; border-radius: 2px; background: {c['RULE']}; vertical-align: middle; }}
.ub-happened.shut span {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip-path: inset(50%); white-space: nowrap; }}
.ub-happened.missed {{ color: {c['VIOLET']}; font-weight: 600; }} .ub-happened.false {{ color: {c['AMBER']}; }}
.gradio-container .bedpick {{ background: {c['TILE']}; border: 1px solid rgba(255,255,255,.07); border-radius: 22px; padding: 12px 14px; gap: 8px; margin-bottom: 14px; }}
.gradio-container .bedpick .buttons {{ flex-wrap: wrap; gap: 8px; }}
.gradio-container .bedpick button {{ flex: 0 0 auto; min-width: 0 !important; width: auto; min-height: 44px; padding: 0 14px; border-radius: 12px; font-size: 15px; font-weight: 600; }}
@media (max-width: 1000px) {{
  .ub-head {{ display: none; }}
  .ub-row {{ grid-template-columns: 40px 1fr 1fr; }}
  .ub-row .ub-level, .ub-row .ub-vitals, .ub-row .ub-happened {{ grid-column: 2 / -1; }}
  .ub-row.away .ub-status {{ grid-column: 3 / -1; }}
}}
@media (max-width: 600px) {{
  .ub-row {{ grid-template-columns: 34px 1fr; padding: 10px; }}
  .ub-row .ub-status, .ub-row .ub-level, .ub-row .ub-vitals, .ub-row .ub-happened {{ grid-column: 2 / -1; }}
  .ub-row.away .ub-status {{ grid-column: 2 / -1; }}
  .ub-rank {{ font-size: 24px; }}
  .bedpick .buttons button {{ flex: 1 1 calc(33% - 8px) !important; }}
}}
"""
