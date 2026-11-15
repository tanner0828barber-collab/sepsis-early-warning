"""Sepsis Early Warning: the replay tool.

Run it:  .venv/bin/python app/app.py

What it does: loads the frozen model, takes one saved, de-identified patient,
and replays the stay hour by hour. At each hour the model is given only the
rows up to that hour and really scores them. Nothing is typed in by the user
and nothing is stored.

The screen is a grid of tiles. Every tile is drawn again at each hour from
what the model could see at that hour.

This project is a prototype and is not medical advice.
"""
import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features  # noqa: E402
from train import score_rows  # noqa: E402

APP_DATA = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------- frozen things
# Before the freeze, the app can preview with the not-yet-frozen candidate.
_frozen_file = ROOT / "results" / "frozen.json"
if not _frozen_file.exists():
    _frozen_file = ROOT / "results" / "a_dev_candidate.json"
FROZEN = json.loads(_frozen_file.read_text())
SMOOTH = FROZEN.get("smooth", 1)
THRESHOLD = FROZEN["models"]["M3"]["threshold"]
COLUMNS = FROZEN["models"]["M3"]["columns"]
MODEL = joblib.load(ROOT / "models" / "M3.joblib")

CASES = json.loads((APP_DATA / "cases.json").read_text())
PATIENTS = pd.read_csv(APP_DATA / "patients.csv")
PATIENTS["hosp"] = "saved"
SCOREBOARD = json.loads((APP_DATA / "scoreboard.json").read_text())
# the "Known weak spots" section is written from the result files by scripts/build_scoreboard.py
LIMITS_TEXT = (Path(__file__).resolve().parent / "limits.md").read_text().replace("RESULTS_PLACEHOLDER", SCOREBOARD["limits"])

# ---------------------------------------------------------------- look and feel
NIGHT, TILE, LANE, RULE = "#050608", "#0E1116", "#141922", "#232B37"
CHALK, FOG = "#E9EEF3", "#8D99A8"
FLARE, VIOLET, TIDE, AMBER = "#FF6B3D", "#D08BFF", "#6FB3E0", "#F6B84B"

# The four vital sign tiles. Names and units are as listed in the dataset's own description on PhysioNet.
# The shaded strips are the scoring ranges of the NEWS2 bedside chart, checked on 11/14/2026 against
# Chart 1 of the Royal College of Physicians NEWS2 report (2017). That chart scores whole numbers, so
# each edge sits half a step between two values. The model does not use these ranges.
# (column, name, unit, color, lowest shown, highest shown, [(from, to, NEWS2 points)])
VITAL_TILES = [
    ("HR", "Heart rate", "beats per minute", "#62E38A", 30, 150,
     [(30, 40.5, 3), (40.5, 50.5, 1), (90.5, 110.5, 1), (110.5, 130.5, 2), (130.5, 150, 3)]),
    ("Temp", "Temperature", "°C", "#F6B84B", 34, 41,
     [(34, 35.05, 3), (35.05, 36.05, 1), (38.05, 39.05, 1), (39.05, 41, 2)]),
    ("Resp", "Breathing rate", "breaths per minute", "#4DD8E6", 4, 36,
     [(4, 8.5, 3), (8.5, 11.5, 1), (20.5, 24.5, 2), (24.5, 36, 3)]),
    ("SBP", "Systolic blood pressure", "mm Hg", "#7AA8FF", 70, 230,
     [(70, 90.5, 3), (90.5, 100.5, 2), (100.5, 110.5, 1), (219.5, 230, 3)]),
]
SEEN_TABLE = [
    ("HR", "Heart rate", "beats per minute", "vital"),
    ("O2Sat", "Oxygen saturation", "%", "vital"),
    ("Temp", "Temperature", "°C", "vital"),
    ("SBP", "Systolic blood pressure", "mm Hg", "vital"),
    ("MAP", "Mean arterial pressure", "mm Hg", "vital"),
    ("DBP", "Diastolic blood pressure", "mm Hg", "vital"),
    ("Resp", "Breathing rate", "breaths per minute", "vital"),
    ("WBC", "White cell count", "thousand per µL", "lab"),
    ("Lactate", "Lactate", "mg/dL", "lab"),
    ("Creatinine", "Creatinine", "mg/dL", "lab"),
    ("Platelets", "Platelets", "thousand per µL", "lab"),
    ("Bilirubin_total", "Total bilirubin", "mg/dL", "lab"),
    ("BUN", "Blood urea nitrogen", "mg/dL", "lab"),
    ("Glucose", "Glucose", "mg/dL", "lab"),
    ("PaCO2", "Carbon dioxide in arterial blood", "mm Hg", "lab"),
]
STALE_AFTER = {"vital": 4, "lab": 24}  # hours; my own choice, not a clinical standard

# Each state has its own shape, color and word, so color is never the only signal.
SHAPES = {
    "quiet": '<circle cx="8" cy="8" r="6" fill="none" stroke="currentColor" stroke-width="2.6"/>',
    "alert": '<path d="M8 1 15.3 14.5H.7Z" fill="currentColor"/>',
    "missed": '<path d="M8 .8 15.2 8 8 15.2.8 8Z" fill="currentColor"/>',
}
PILL_SHAPE = {"caught": "alert", "false alarm": "alert", "missed": "missed"}

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@600;700;800&family=IBM+Plex+Sans:wght@400;600&display=swap');
body, gradio-app {{ background: {NIGHT} !important; }}
.gradio-container, .gradio-container * {{ font-family: 'IBM Plex Sans', system-ui, sans-serif !important; }}
.gradio-container {{
  background: {NIGHT} !important; color: {CHALK}; max-width: 1180px !important; margin: 0 auto !important; padding: 18px 20px 40px !important;
  font-size: 16px; line-height: 1.5;
  --body-background-fill: {NIGHT}; --background-fill-primary: {NIGHT}; --background-fill-secondary: {TILE};
  --block-background-fill: transparent; --block-border-color: transparent; --border-color-primary: {RULE}; --border-color-accent: {CHALK};
  --body-text-color: {CHALK}; --body-text-color-subdued: {FOG}; --block-label-text-color: {FOG}; --block-title-text-color: {FOG};
  --block-label-background-fill: transparent; --block-title-background-fill: transparent;
  --input-background-fill: {LANE}; --input-border-color: {RULE}; --input-placeholder-color: {FOG};
  --button-secondary-background-fill: {LANE}; --button-secondary-background-fill-hover: {RULE}; --button-secondary-text-color: {CHALK};
  --button-secondary-text-color-hover: {CHALK}; --button-secondary-border-color: transparent; --button-secondary-border-color-hover: {FOG};
  --button-primary-background-fill: {CHALK}; --button-primary-background-fill-hover: #FFFFFF; --button-primary-text-color: {NIGHT};
  --button-primary-text-color-hover: {NIGHT}; --button-primary-border-color: {CHALK}; --button-primary-border-color-hover: #FFFFFF;
  --slider-color: {CHALK}; --color-accent: {CHALK}; --color-accent-soft: {LANE}; --link-text-color: {TIDE}; --link-text-color-hover: {TIDE};
  --link-text-color-visited: {TIDE}; --link-text-color-active: {TIDE}; --table-border-color: {RULE}; --table-even-background-fill: transparent;
  --table-odd-background-fill: transparent; --neutral-200: {RULE}; --neutral-700: {RULE}; --neutral-800: {TILE}; --shadow-drop: none; --shadow-drop-lg: none;
  --block-shadow: none; --layout-gap: 14px;
}}
.gradio-container .prose {{ font-size: 16px; line-height: 1.5; }}
.gradio-container .display, .gradio-container .display * {{ font-family: 'Sora', 'IBM Plex Sans', system-ui, sans-serif !important; }}
.gradio-container :is(button, a, input, [tabindex]):focus-visible {{ outline: 3px solid {TIDE} !important; outline-offset: 2px; }}
footer {{ display: none !important; }}

.topline {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px 16px; }}
.topline .name {{ font-weight: 800; font-size: 22px; letter-spacing: -.01em; color: {CHALK}; }}
.topline .tag {{ color: {FOG}; font-size: 14px; border: 1px solid {RULE}; border-radius: 999px; padding: 4px 12px; }}
.preview {{ background: rgba(246,184,75,.16); color: {AMBER}; border-radius: 12px; padding: 8px 14px; font-weight: 600; }}

/* the three tabs, drawn as one rounded switch */
.gradio-container .tab-nav {{ display: inline-flex; gap: 4px; background: {TILE}; border: 1px solid rgba(255,255,255,.07) !important; border-radius: 999px; padding: 4px; margin: 6px 0 14px; width: fit-content; max-width: 100%; overflow-x: auto; flex-wrap: nowrap; }}
.gradio-container .tab-nav button {{ color: {FOG}; background: transparent; border: 0 !important; border-radius: 999px !important; padding: 8px 16px; font-weight: 600; font-size: 16px; white-space: nowrap; }}
.gradio-container .tab-nav button.selected {{ color: {NIGHT}; background: {CHALK}; }}
.gradio-container .tabitem {{ border: 0 !important; padding: 0 !important; background: transparent !important; }}

/* tiles */
.tile {{ background: {TILE}; border: 1px solid rgba(255,255,255,.07); border-radius: 22px; padding: 18px 20px; min-width: 0; color: {CHALK}; }}
.tile h3 {{ margin: 0; font-size: 14px; font-weight: 600; color: {FOG}; display: flex; align-items: center; gap: 8px; line-height: 1.4; }}
.grid12 {{ display: grid; grid-template-columns: repeat(12, 1fr); gap: 14px; }}
.caption {{ font-size: 14px; color: {FOG}; line-height: 1.5; margin: 0; }}
.source {{ color: {FOG}; font-size: 14px; line-height: 1.5; margin: 16px 2px 0; max-width: 105ch; }}

/* the controls tile: saved patients, the hour slider, Play and Step */
.gradio-container .transport {{ background: {TILE}; border: 1px solid rgba(255,255,255,.07); border-radius: 22px; padding: 12px 14px 10px; gap: 8px; margin-bottom: 14px; }}
.gradio-container .transport .buttons {{ flex-wrap: wrap; gap: 10px; align-items: center; }}
.gradio-container .transport button {{ flex: 0 0 auto; min-width: 0 !important; width: auto; min-height: 46px; padding: 0 16px; border-radius: 14px; font-size: 16px; font-weight: 600; border: 1px solid transparent; }}
.gradio-container .transport button.pill.primary {{ background: {LANE}; color: {CHALK}; border-color: {CHALK}; }}
.gradio-container .transport button.pill::before {{ content: ""; width: 12px; height: 12px; margin-right: 8px; flex: none; }}
.gradio-container .transport button.pill.alert::before {{ background: {FLARE}; clip-path: polygon(50% 0, 100% 100%, 0 100%); }}
.gradio-container .transport button.pill.missed::before {{ background: {VIOLET}; clip-path: polygon(50% 0, 100% 50%, 50% 100%, 0 50%); }}
.gradio-container .transport button.go {{ padding: 0 20px; }}
.gradio-container .transport button.go.secondary {{ background: transparent; border-color: {RULE}; }}
.gradio-container .transport .spacer {{ flex: 1 1 0; min-width: 0 !important; }}
.gradio-container .transport .form {{ background: transparent; border: 0; box-shadow: none; }}
.gradio-container .transport .hour {{ padding: 0 4px !important; border: 0 !important; background: transparent !important; }}
.gradio-container .transport .hour label span, .gradio-container .transport .hour .head span {{ color: {FOG}; font-size: 14px; font-weight: 600; }}
.gradio-container .transport .hour input[type=number] {{ background: {LANE}; color: {CHALK}; border: 1px solid {RULE}; border-radius: 10px; font-weight: 600; font-size: 16px; }}
.gradio-container .transport .hour input[type=range] {{ accent-color: {CHALK}; }}

/* the state tile */
.state {{ grid-column: span 4; grid-row: span 2; display: flex; flex-direction: column; justify-content: flex-end; min-height: 330px; position: relative; overflow: hidden; }}
.state .corner {{ position: absolute; top: 18px; left: 20px; right: 20px; display: flex; justify-content: space-between; font-size: 14px; font-weight: 600; }}
.state .corner span {{ color: #C5CED8; }}
.state .word {{ font-weight: 800; font-size: clamp(44px, 5.4vw, 68px); line-height: 1; letter-spacing: -.03em; display: flex; align-items: center; gap: .16em; margin-bottom: 10px; }}
.state .word svg {{ width: .66em; height: .66em; flex: none; }}
.state .lead {{ font-size: 20px; font-weight: 600; margin: 0 0 6px; }}
.state p {{ margin: 0 0 6px; }}
.state.alert {{ background: radial-gradient(120% 90% at 0% 0%, rgba(255,107,61,.34), rgba(255,107,61,.05) 62%), {TILE}; border-color: rgba(255,107,61,.5); }}
.state.alert .word, .state.alert .word * {{ color: {FLARE}; }}
.state.missed {{ background: radial-gradient(120% 90% at 0% 0%, rgba(208,139,255,.30), rgba(208,139,255,.05) 62%), {TILE}; border-color: rgba(208,139,255,.5); }}
.state.missed .word, .state.missed .word * {{ color: {VIOLET}; }}
.state.quiet .word, .state.quiet .word * {{ color: {CHALK}; }}

/* the four vital sign tiles */
.vital {{ grid-column: span 4; display: flex; flex-direction: column; gap: 6px; }}
.vital h3 i {{ width: 9px; height: 9px; border-radius: 50%; background: var(--c); }}
.vital .val {{ display: flex; align-items: flex-end; gap: 12px; }}
.vital .val b {{ font-weight: 700; font-size: 52px; line-height: 1; letter-spacing: -.03em; color: var(--c); }}
.vital .val span {{ color: {FOG}; font-size: 13.5px; line-height: 1.35; padding-bottom: 4px; }}
.vital .val b.none {{ font-size: 20px; line-height: 52px; color: {FOG}; letter-spacing: 0; font-weight: 400; }}
.vital svg {{ width: 100%; height: auto; display: block; border-radius: 8px; background: {LANE}; margin-top: auto; }}

/* the score tile */
.scoretile {{ grid-column: span 12; }}
.scorehead {{ display: flex; flex-wrap: wrap; align-items: flex-end; gap: 6px 16px; margin: 4px 0 6px; }}
.scorehead b {{ font-weight: 700; font-size: 52px; line-height: 1; letter-spacing: -.03em; }}
.scorehead b.hot {{ color: {FLARE}; }}
.scorehead span {{ color: {FOG}; padding-bottom: 5px; }}
.scoretile svg {{ width: 100%; height: auto; }}
.tile svg.wide {{ display: block; }} .tile svg.narrow {{ display: none; }}
svg .lane {{ fill: {LANE}; }} svg .over {{ fill: rgba(255,107,61,.09); }} svg .window {{ fill: rgba(233,238,243,.09); }}
svg .grid {{ stroke: {RULE}; stroke-width: 1; }} svg .alertline {{ stroke: {FLARE}; stroke-width: 1.6; stroke-dasharray: 7 5; }}
svg .scoreline {{ fill: none; stroke: {CHALK}; stroke-width: 2.6; stroke-linejoin: round; stroke-linecap: round; }}
svg .hotdot {{ fill: {FLARE}; stroke: {NIGHT}; stroke-width: 1; }} svg .nowdot {{ fill: {CHALK}; stroke: {NIGHT}; stroke-width: 1.5; }}
svg .tick {{ fill: {FOG}; font-size: 12.5px; }} svg .alerttext {{ fill: {FLARE}; font-size: 12.5px; }}
svg .band1 {{ fill: rgba(233,238,243,.05); }} svg .band2 {{ fill: rgba(233,238,243,.10); }} svg .band3 {{ fill: rgba(233,238,243,.16); }}
svg .halo {{ paint-order: stroke; stroke: {TILE}; stroke-width: 5px; stroke-linejoin: round; font-size: 12.5px; }}
svg .mk-label {{ stroke: {FOG}; stroke-width: 1.4; stroke-dasharray: 2 4; }} svg .mk-label-t {{ fill: {FOG}; }}
svg .mk-alert {{ stroke: {FLARE}; stroke-width: 2; }} svg .mk-alert-t {{ fill: {FLARE}; font-weight: 600; }} svg .mk-alert-head {{ fill: {FLARE}; }}
svg .mk-onset {{ stroke: {CHALK}; stroke-width: 2; }} svg .mk-onset-t {{ fill: {CHALK}; font-weight: 600; }}

/* what happened, and the hospital B numbers */
.happened {{ grid-column: span 7; display: flex; flex-direction: column; }}
.happened .head {{ font-weight: 700; font-size: 34px; line-height: 1.1; letter-spacing: -.02em; margin: 8px 0 4px; }}
.happened p {{ margin: 0 0 6px; max-width: 56ch; }}
.happened svg {{ width: 100%; height: auto; margin-top: auto; }}
.happened.shut {{ border-style: dashed; border-color: {RULE}; }} .happened.shut .head {{ color: {FOG}; }}
.proof {{ grid-column: span 5; }}
.proof .lede {{ margin: 6px 0 10px; }}
.nums {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px 16px; }}
.nums div b {{ display: block; font-weight: 700; font-size: 30px; line-height: 1.1; letter-spacing: -.02em; font-variant-numeric: tabular-nums; }}
.nums div.first {{ grid-column: span 3; }} .nums div.first b {{ font-size: 40px; }}
.nums span {{ color: {FOG}; font-size: 14px; line-height: 1.35; display: block; }}

/* every input the model could see */
.inputs {{ grid-column: span 12; }}
.cells {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(172px, 1fr)); gap: 10px; margin: 10px 0; }}
.cell {{ background: {LANE}; border-radius: 14px; padding: 10px 12px; display: flex; flex-direction: column; gap: 1px; }}
.cell span {{ color: {FOG}; font-size: 13px; }}
.cell b {{ font-size: 19px; font-weight: 600; font-variant-numeric: tabular-nums; }}
.cell small {{ color: {FOG}; font-weight: 400; font-size: 13px; }}
.cell b.none {{ color: {FOG}; font-size: 15px; font-weight: 400; }}
.cell em {{ font-style: normal; background: rgba(246,184,75,.16); color: {AMBER}; border-radius: 4px; padding: 0 6px; margin-left: 4px; }}
.wide12 {{ grid-column: span 12; }}

/* the "Does it travel?" tab */
.headline {{ font-weight: 700; font-size: 30px; line-height: 1.15; letter-spacing: -.02em; margin: 2px 0 10px; max-width: 26ch; }}
.finding {{ font-size: 18px; line-height: 1.5; margin: 0; max-width: 78ch; }}
.tile .scroll {{ overflow-x: auto; margin-top: 10px; }}
.gradio-container table.plain {{ border-collapse: collapse; width: 100%; font-size: 16px; border: 0 !important; margin: 0; }}
.gradio-container table.plain :is(th, td) {{ border: 0 !important; border-bottom: 1px solid {RULE} !important; padding: 8px; background: transparent !important; }}
.gradio-container table.plain tr {{ border: 0 !important; }}
.gradio-container table.plain th {{ text-align: left; font-weight: 600; color: {FOG}; font-size: 14px; vertical-align: bottom; }}
.gradio-container table.plain td {{ color: {CHALK}; font-variant-numeric: tabular-nums; vertical-align: top; }}
.gradio-container table.plain tr:last-child td {{ border-bottom: 0 !important; }}
.gradio-container table.plain :is(td.num, th.num) {{ text-align: right; }}
.gradio-container table.plain.many {{ min-width: 860px; }}
.tile .scroll + .caption, .tile .barkey + .caption {{ margin-top: 10px; }}
.range {{ color: {FOG}; font-size: 14px; }}
.charttitle {{ font-weight: 700; font-size: 22px; line-height: 1.2; letter-spacing: -.01em; margin: 8px 0 14px; max-width: 40ch; color: {CHALK}; }}
.bars {{ display: flex; align-items: flex-end; gap: 10px; height: 250px; }}
.bars .bar {{ flex: 1; display: flex; flex-direction: column; justify-content: flex-end; align-items: center; height: 100%; min-width: 0; }}
.bars .bar b {{ font-weight: 700; font-size: 20px; }}
.bars .bar i {{ display: block; width: 100%; background: #56616F; border-radius: 8px 8px 2px 2px; min-height: 2px; }}
.bars .bar.timely i {{ background: {CHALK}; }}
.barlabels {{ display: flex; gap: 10px; margin-top: 6px; }}
.barlabels span {{ flex: 1; text-align: center; color: {FOG}; font-size: 13.5px; line-height: 1.3; min-width: 0; }}
.barkey {{ display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 14px; margin: 12px 0 8px; }}
.barkey span {{ color: {FOG}; }}
.barkey i {{ display: inline-block; width: 14px; height: 14px; border-radius: 4px; vertical-align: -2px; margin-right: 6px; }}
.box {{ font-size: 16px; line-height: 1.55; max-width: 90ch; }}
.tile.after {{ border-style: dashed; border-color: {RULE}; }}
.tile.after h3 em {{ font-style: normal; background: rgba(246,184,75,.16); color: {AMBER}; border-radius: 6px; padding: 1px 8px; white-space: nowrap; }}
.tile.after svg {{ width: 100%; height: auto; }}
.tile.after .ytitle {{ margin: 0 0 4px; }} .tile.after svg + .caption {{ margin-top: 10px; max-width: 110ch; }}
svg .value {{ fill: {CHALK}; font-size: 13px; font-weight: 600; }} svg .alonedot {{ fill: {TILE}; stroke: {FOG}; stroke-width: 2; }}

/* the limits tab */
.gradio-container .limits {{ background: {TILE}; border: 1px solid rgba(255,255,255,.07); border-radius: 22px; padding: 22px 26px; }}
.gradio-container .limits .prose {{ max-width: 80ch; color: {CHALK}; font-size: 16px; background: transparent; border: 0; padding: 0; }}
.gradio-container .limits .prose :is(h1, h2, h3, strong, th, td, li, p) {{ color: {CHALK}; }}
.gradio-container .limits .prose h1 {{ font-size: 30px; letter-spacing: -.02em; }}
.gradio-container .limits .prose h2 {{ font-size: 20px; margin-top: 26px; }}
.gradio-container .limits .prose :is(th, td) {{ border-color: {RULE} !important; background: transparent !important; }}
.gradio-container .limits .prose table {{ display: block; overflow-x: auto; }}
.gradio-container .limits .prose tr {{ border-color: {RULE} !important; }}

@media (max-width: 900px) {{
  .state {{ grid-column: span 12; min-height: 0; padding-top: 52px; }} .state .word {{ font-size: 64px; }} .vital {{ grid-column: span 6; }} .happened, .proof {{ grid-column: span 12; }}
}}
/* Gradio adds its own prefix to rules inside a media block, so these must not start with .gradio-container */
@media (max-width: 600px) {{
  .grid12 {{ gap: 10px; }} .tile {{ border-radius: 18px; padding: 14px; }}
  .tile svg.wide {{ display: none; }} .tile svg.narrow {{ display: block; }} .state {{ padding-top: 46px; }}
  .tabs .tab-nav {{ border-radius: 22px; flex-wrap: wrap; }}
  .state .word {{ font-size: 56px; }}
  .vital .val {{ flex-direction: column; align-items: flex-start; gap: 2px; }} .vital .val b {{ font-size: 40px; }} .vital .val b.none {{ line-height: 40px; }}
  .transport .buttons button.pill {{ flex: 1 1 calc(50% - 5px) !important; width: auto !important; padding: 0 10px; }}
  .transport .buttons button.go {{ flex: 1 1 calc(50% - 5px) !important; width: auto !important; }}
  .transport .buttons .spacer {{ display: none; }}
  .nums {{ grid-template-columns: 1fr 1fr; }} .nums div.first {{ grid-column: span 2; }} .nums div.first b {{ font-size: 34px; }}
  .cells {{ grid-template-columns: 1fr 1fr; }} .headline {{ font-size: 24px; }} .finding {{ font-size: 16px; }}
  .bars {{ gap: 4px; height: 200px; }} .barlabels {{ gap: 4px; }} .bars .bar b {{ font-size: 15px; }} .barlabels span {{ font-size: 11.5px; }}
  .tabitem .limits {{ padding: 16px; }}
}}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; scroll-behavior: auto !important; }} }}
"""

# The page is drawn dark whatever the visitor's system setting is.
FORCE_DARK = "() => { document.body.classList.add('dark'); document.documentElement.style.colorScheme = 'dark'; }"


# ---------------------------------------------------------------- the model at one hour
def record(pid):
    """Every recorded hour for one saved patient, in time order."""
    return PATIENTS[PATIENTS.pid == pid].reset_index(drop=True)


@lru_cache(maxsize=4096)
def seen_so_far(pid, hour):
    """Give the model only the rows up to `hour` and score them.

    Returns the inputs and scores for every hour up to and including `hour`.
    The model never receives a later row.
    """
    rows = record(pid)
    rows = rows[rows.ICULOS <= hour]
    inputs = build_features(rows)
    inputs["score"] = score_rows(MODEL, inputs, COLUMNS, SMOOTH)
    return inputs


def truth(pid):
    """What the record says happened. Shown only once the replay gets there."""
    rows = record(pid)
    on = rows[rows.SepsisLabel == 1]
    if on.empty:
        return {"sepsis": False, "label_start": None, "onset": None}
    label_start = int(on.ICULOS.iloc[0])
    return {
        "sepsis": True,
        "label_start": label_start,
        "onset": label_start + 6,
        "late_entry": bool(rows.SepsisLabel.iloc[0] == 1),
    }


def hours_of(pid):
    rows = record(pid)
    return int(rows.ICULOS.iloc[0]), int(rows.ICULOS.iloc[-1])


def plural(n, word):
    n = int(n)
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def moment(pid, hour):
    """Everything the tiles need for one patient at one hour.

    The scores come from seen_so_far, so they use only rows up to `hour`.
    The outcome fields (onset, miss, hours of warning) are worked out here but
    each tile shows them only when `revealed` or `ended` says the replay has got there.
    """
    first, last = hours_of(pid)
    seen = seen_so_far(pid, hour)
    rows = record(pid)
    rows = rows[rows.ICULOS <= hour]
    facts = truth(pid)
    relative = (seen["score"] / THRESHOLD).tolist()   # the score as a multiple of the frozen alert line
    alerting = seen[seen["score"] >= THRESHOLD]
    first_alert = int(alerting.ICULOS.iloc[0]) if not alerting.empty else None
    revealed = bool(facts["sepsis"] and facts["onset"] <= hour)
    in_time = bool(facts["sepsis"] and first_alert is not None and first_alert <= facts["onset"])
    return SimpleNamespace(
        pid=pid, hour=hour, first=first, last=last, span=max(last - first, 1), seen=seen, rows=rows,
        hours=seen.ICULOS.tolist(), relative=relative, now=seen.iloc[-1], now_relative=relative[-1],
        sepsis=facts["sepsis"], label_start=facts["label_start"], onset=facts["onset"],
        first_alert=first_alert, revealed=revealed, missed=revealed and not in_time,
        warned=(facts["onset"] - first_alert) if revealed and in_time else None, ended=hour >= last,
    )


def marker_list(m):
    """The three markers, each shown only once the replay reaches it."""
    marks = []
    if m.first_alert is not None:
        marks.append((m.first_alert, "First alert", "mk-alert", True))
    if m.revealed:
        marks.append((m.onset, "Onset", "mk-onset", True))
    if m.sepsis and m.label_start <= m.hour:
        marks.append((m.label_start, "Label start", "mk-label", False))
    return marks


# ---------------------------------------------------------------- the tiles
def shape(kind):
    return f'<svg viewBox="0 0 16 16" aria-hidden="true">{SHAPES[kind]}</svg>'


def state_tile(m, label):
    """Quiet, Alert or Missed: a shape, a color and a word together."""
    if m.missed:
        kind, word, lead = "missed", "Missed", "No alert by onset."
        text = [f"Sepsis onset was at hour {m.onset}. The score never crossed the alert line in time."]
        if m.first_alert is not None:
            text.append(f"The first alert came at hour {m.first_alert}, after onset.")
    elif m.first_alert is None:
        kind, word, lead = "quiet", "Quiet", "No alert."
        text = ["The score has stayed under the alert line."]
    else:
        kind, word, lead = "alert", "Alert", "Look at this patient first."
        if m.warned is not None:
            text = [f"First fired at hour {m.first_alert}, {plural(m.warned, 'hour')} before sepsis onset at hour {m.onset}."]
        else:
            text = [f"First fired at hour {m.first_alert}."]
        if m.now_relative < 1:
            text.append("The score is back under the alert line now.")
    return (
        f'<section class="tile state {kind}"><div class="corner"><span>{label}</span><span>Hour {m.hour} of {m.last}</span></div>'
        f'<div role="status"><div class="word display">{shape(kind)}<span>{word}</span></div>'
        f'<p class="lead">{lead}</p>{"".join(f"<p>{t}</p>" for t in text)}</div></section>'
    )


def vital_tile(m, column, name, unit, color, low, high, bands):
    """One vital sign: the latest value the model could see, over a small chart of the stay so far."""
    width, height = 360, 70
    x = lambda h: 4 + (h - m.first) / m.span * (width - 8)
    y = lambda v: 3 + (high - min(max(v, low), high)) / (high - low) * (height - 6)
    parts = [f'<rect x="0" y="{y(b):.1f}" width="{width}" height="{y(a) - y(b):.1f}" class="band{points}"/>' for a, b, points in bands]
    if m.first_alert is not None:
        parts.append(f'<line x1="{x(m.first_alert):.1f}" x2="{x(m.first_alert):.1f}" y1="0" y2="{height}" class="mk-alert" opacity=".7"/>')
    if m.revealed:
        parts.append(f'<line x1="{x(m.onset):.1f}" x2="{x(m.onset):.1f}" y1="0" y2="{height}" class="mk-onset" opacity=".7"/>')
    # a dot is a real measurement; the line carries the last one forward
    path, end = "", None
    for h, v in zip(m.hours, m.seen[f"last_{column}"]):
        if pd.isna(v):
            continue
        path += f"M{x(h):.1f},{y(v):.1f} " if not path else f"H{x(h):.1f} V{y(v):.1f} "
        end = (x(h), y(v))
    if path:
        parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.4" opacity=".55"/>')
    for h, v in zip(m.rows.ICULOS, m.rows[column]):
        if not pd.isna(v):
            parts.append(f'<circle cx="{x(h):.1f}" cy="{y(v):.1f}" r="2.1" fill="{color}"><title>Hour {int(h)}: {v:g} {unit} (measured)</title></circle>')
    if end:
        parts.append(f'<circle cx="{end[0]:.1f}" cy="{end[1]:.1f}" r="4" fill="{color}" stroke="{TILE}" stroke-width="1.5"/>')
    value, age = m.now[f"last_{column}"], m.now[f"hours_since_{column}"]
    if pd.isna(value):
        big, when = '<b class="none">not measured yet</b>', ""
    else:
        big = f'<b class="display">{value:.1f}</b>' if column == "Temp" else f'<b class="display">{value:.0f}</b>'
        when = f'<span>{unit}<br>measured {"this hour" if age == 0 else plural(age, "hour") + " ago"}</span>'
    return (
        f'<section class="tile vital" style="--c:{color}"><h3><i></i>{name}</h3><div class="val">{big}{when}</div>'
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{name}, hours {m.first} to {m.hour}">{"".join(parts)}</svg></section>'
    )


def score_chart(m, narrow=False):
    """The model score over the stay so far, as a multiple of the frozen alert line, with the markers.

    `narrow` is the phone layout: the same data with fewer pixels across.
    """
    width, left, right = (380, 30, 8) if narrow else (1100, 40, 14)
    plot, font = width - left - right, 12.5
    rail = [14, 30, 46]                     # one row per marker word, so two words never share a line
    top, tall = 56, 150 if narrow else 170
    bottom = top + tall
    x = lambda h: left + (h - m.first) / m.span * plot
    low, high = math.log10(0.03), math.log10(30)   # log scale from 0.03 to 30 times the alert line
    y = lambda v: top + (high - math.log10(min(max(v, 0.03), 30))) / (high - low) * tall
    parts = [
        '<defs><linearGradient id="under" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#E9EEF3" stop-opacity=".16"/>'
        '<stop offset="1" stop-color="#E9EEF3" stop-opacity="0"/></linearGradient></defs>',
        f'<rect x="{left}" y="{top}" width="{plot}" height="{tall}" rx="8" class="lane"/>',
        f'<rect x="{left}" y="{top}" width="{plot}" height="{y(1) - top:.1f}" class="over"/>',
    ]
    if m.revealed:   # the 12 hours before onset, where a first alert counts as timely
        start = max(x(m.onset - 12), left)
        parts.append(f'<rect x="{start:.1f}" y="{top}" width="{x(m.onset) - start:.1f}" height="{tall}" class="window"/>')
    for value, text in [(10, "10"), (1, "1"), (0.1, "0.1")]:
        parts.append(
            f'<line x1="{left}" x2="{left + plot}" y1="{y(value):.1f}" y2="{y(value):.1f}" class="{"alertline" if value == 1 else "grid"}"/>'
            f'<text x="{left - 6}" y="{y(value) + 4:.1f}" class="tick" text-anchor="end">{text}</text>'
        )
    parts.append(f'<text x="{left + 8}" y="{y(1) - 6:.1f}" class="alerttext">Alert line, frozen on hospital A</text>')
    points = [(x(h), y(v)) for h, v in zip(m.hours, m.relative)]
    if len(points) > 1:
        line = " ".join(f"{a:.1f},{b:.1f}" for a, b in points)
        parts.append(f'<polygon points="{points[0][0]:.1f},{bottom} {line} {points[-1][0]:.1f},{bottom}" fill="url(#under)"/>')
        parts.append(f'<polyline points="{line}" class="scoreline"/>')
    for h, v in zip(m.hours, m.relative):
        if v >= 1:
            parts.append(f'<circle cx="{x(h):.1f}" cy="{y(v):.1f}" r="3.4" class="hotdot"><title>Hour {int(h)}: {v:.2f} times the alert line</title></circle>')
    parts.append(f'<circle cx="{points[-1][0]:.1f}" cy="{points[-1][1]:.1f}" r="4.6" class="nowdot"/>')
    every = next(s for s in (1, 2, 5, 10, 20, 50, 100, 200, 500) if m.span / s <= (5 if narrow else 14))
    for h in range((m.first // every + 1) * every, m.last + 1, every):
        parts.append(f'<text x="{x(h):.1f}" y="{bottom + 17}" class="tick" text-anchor="middle">{h}</text>')
    parts.append(f'<text x="{left + plot}" y="{bottom + 36}" class="tick" text-anchor="end">Hours in the unit</text>')

    # markers: a line down the chart and a word on its own row above it. A word goes on the side of
    # its line where it fits and crosses the fewest other lines.
    poles = []
    for row, (h, word, style, bold) in enumerate(marker_list(m)):
        at, base, room = x(h), rail[row], len(word) * font * (0.62 if bold else 0.56)
        sides = [("start", at + 7), ("end", at - 7 - room)]
        if style == "mk-label":
            sides.reverse()
        best = None
        for anchor, begin in sides:
            if begin < 2 or begin + room > width - 2:
                continue
            crossed = sum(1 for pole, pole_row in poles if pole_row < row and begin - 3 < pole < begin + room + 3)
            if best is None or crossed < best[0]:
                best = (crossed, anchor)
        anchor = best[1] if best else "end"
        parts.append(f'<line x1="{at:.1f}" x2="{at:.1f}" y1="{base - 9}" y2="{bottom}" class="{style}"/>')
        if style == "mk-alert":
            parts.append(f'<path d="M{at - 5.5:.1f},{base - 1} L{at:.1f},{base - 11} L{at + 5.5:.1f},{base - 1}Z" class="mk-alert-head"/>')
        parts.append(f'<text x="{at + (7 if anchor == "start" else -7):.1f}" y="{base}" class="{style}-t halo" text-anchor="{anchor}">{word}</text>')
        poles.append((at, row))
    label = f"Model score, hours {m.first} to {m.hour}. It is {m.now_relative:.2f} times the alert line now."
    return f'<svg class="{"narrow" if narrow else "wide"}" viewBox="0 0 {width} {bottom + 42}" role="img" aria-label="{label}">{"".join(parts)}</svg>'


def score_tile(m):
    return (
        '<section class="tile scoretile"><h3>Model score divided by the frozen alert cutoff. Not a probability.</h3>'
        f'<div class="scorehead"><b class="display{" hot" if m.now_relative >= 1 else ""}">{m.now_relative:.2f}</b>'
        f'<span>times the alert line at hour {m.hour}</span></div>{score_chart(m)}{score_chart(m, narrow=True)}</section>'
    )


def happened_tile(m):
    """What really happened. Filled in only when the replay reaches it."""
    if m.revealed and not m.missed:
        head = "Alert at the onset hour" if m.warned == 0 else f"{plural(m.warned, 'hour')} of warning"
        body = (
            "A timely catch. The first alert came in the 12 hours before onset." if m.warned <= 12
            else "Alerted very early. More than 12 hours before onset is counted separately from a timely catch."
        )
    elif m.revealed:
        head, body = "Missed", f"Sepsis onset was at hour {m.onset}. The score never crossed the alert line in time. This is a miss."
    elif not m.sepsis and m.ended:
        head = "No sepsis, no alert" if m.first_alert is None else "False alarm"
        body = "End of the record. This patient never developed sepsis" + (
            ", and there was no alert." if m.first_alert is None else ". This alert was a false alarm."
        )
    elif m.sepsis and m.ended:
        head, body = "Record ended", "The record ends before the onset hour."
    else:
        head, body = "Not revealed yet", "The outcome shows only when the replay reaches it. The model never sees it."

    shut = head == "Not revealed yet"
    return (
        f'<section class="tile happened {"shut" if shut else "open"}"><h3>What happened</h3>'
        f'<div class="head display">{head}</div><p>{body}</p>{stay_strip(m)}{stay_strip(m, narrow=True)}</section>'
    )


def stay_strip(m, narrow=False):
    """One line for the whole stay: how far the replay has got, and where the alert and the onset fell."""
    width, base = 330 if narrow else 600, 44
    x = lambda h: 10 + (h - m.first) / m.span * (width - 20)
    parts = [
        f'<line x1="10" x2="{width - 10}" y1="{base}" y2="{base}" class="grid" style="stroke-width:2"/>',
        f'<line x1="10" x2="{x(m.hour):.1f}" y1="{base}" y2="{base}" stroke="{CHALK}" stroke-width="2"/>',
    ]
    if m.revealed:
        start = max(x(m.onset - 12), 10)
        parts.append(f'<rect x="{start:.1f}" y="{base - 12}" width="{x(m.onset) - start:.1f}" height="24" rx="3" class="window"/>')
    marks = sorted((h, word, style) for h, word, style, _ in marker_list(m) if style != "mk-label")
    for i, (h, word, style) in enumerate(marks):   # the earlier word sits above the line, the later one below
        at, anchor = x(h), "end" if x(h) > width / 2 else "start"
        parts.append(f'<line x1="{at:.1f}" x2="{at:.1f}" y1="{base - 14}" y2="{base + 14}" class="{style}" style="stroke-width:2.5"/>')
        parts.append(
            f'<text x="{at + (-6 if anchor == "end" else 6):.1f}" y="{base - 20 if i == 0 else base + 31}" class="{style}-t" '
            f'font-size="13" text-anchor="{anchor}">{word}, hour {h}</text>'
        )
    return (
        f'<svg class="{"narrow" if narrow else "wide"}" viewBox="0 0 {width} 84" role="img" '
        f'aria-label="Timeline of the stay up to hour {m.hour}">{"".join(parts)}</svg>'
    )


def proof_tile():
    """The hospital B numbers, from the result files. The timely number leads."""
    proof = SCOREBOARD["proof"]
    cells = "".join(
        f'<div class="{"first" if i == 0 else ""}"><b class="display">{c["big"]}</b><span>{c["what"]}</span></div>'
        for i, c in enumerate(proof["numbers"])
    )
    return f'<section class="tile proof"><h3>{proof["title"]}</h3><p class="lede">{proof["lede"]}</p><div class="nums">{cells}</div></section>'


def inputs_tile(m):
    cells = []
    for column, name, unit, kind in SEEN_TABLE:
        value, age = m.now[f"last_{column}"], m.now[f"hours_since_{column}"]
        if pd.isna(value):
            cells.append(f'<div class="cell"><span>{name}</span><b class="none">not measured</b></div>')
            continue
        when = "this hour" if age == 0 else f"{plural(age, 'hour')} ago"
        stale = " <em>stale</em>" if age > STALE_AFTER[kind] else ""
        cells.append(f'<div class="cell"><span>{name}</span><b>{value:g} <small>{unit}</small></b><span>measured {when}{stale}</span></div>')
    return (
        f'<section class="tile inputs"><h3>What the model could see at hour {m.hour}</h3><div class="cells">{"".join(cells)}</div>'
        '<p class="caption">"Stale" means a vital sign older than 4 hours or a lab older than 24 hours. '
        "That cutoff is my own choice for this display. Units are as listed in the dataset's description. "
        "The model also sees trends, time in the unit, and how long ago each value was measured.</p></section>"
    )


LABELS = {case["pid"]: case["label"] for case in CASES["cases"]}
SITE = CASES["site"][0].lower() + CASES["site"][1:]   # "hospital B", for the middle of a sentence


def show(pid, hour):
    """Every tile on the Replay tab for one patient at one hour."""
    if not pid:
        return (
            '<div class="grid12"><section class="tile wide12"><div class="head display">Choose a saved patient above to start the replay.</div>'
            f"</section>{proof_tile()}</div>"
        )
    first, last = hours_of(pid)
    m = moment(pid, int(min(max(hour, first), last)))
    vitals = "".join(vital_tile(m, *tile) for tile in VITAL_TILES)
    return (
        f'<div class="grid12">{state_tile(m, LABELS[pid])}{vitals}{score_tile(m)}{happened_tile(m)}{proof_tile()}{inputs_tile(m)}</div>'
        f'<p class="source">Source: PhysioNet / Computing in Cardiology Challenge 2019, {SITE}, one public de-identified record. '
        "In the four small charts a dot is a real measurement, the line carries the last one forward, and the gray strips are where the "
        "NEWS2 bedside chart (Royal College of Physicians, 2017, made for general wards) gives 1, 2 or 3 points. The model does not use NEWS2. "
        "Retrospective replay of a saved record, not a live monitor. Nothing you do here is stored. "
        "This project is a prototype and is not medical advice.</p>"
    )


# ---------------------------------------------------------------- the "Does it travel?" tab
def html_table(table, many=False):
    head = "".join(
        f'<th class="{"num" if i else ""}">{h}</th>' for i, h in enumerate(table["columns"])
    )
    body = ""
    for row in table["rows"]:
        cells = "".join(
            f'<td class="{"num" if i else ""}">{cell}</td>' for i, cell in enumerate(row)
        )
        body += f"<tr>{cells}</tr>"
    caption = f'<p class="caption">{table["caption"]}</p>' if table.get("caption") else ""
    return (
        f'<div class="scroll"><table class="plain{" many" if many else ""}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>{caption}'
    )


def warning_chart():
    """Hours of warning, as plain bars that fill the tile at any width."""
    data = SCOREBOARD.get("warning_chart")
    if not data:
        return ""
    tallest = max(data["counts"])
    timely = data.get("timely", [False] * len(data["counts"]))
    bars = "".join(
        f'<div class="bar{" timely" if inside else ""}"><b class="display">{count}</b><i style="height:{100 * count / tallest * 0.86:.1f}%"></i></div>'
        for count, inside in zip(data["counts"], timely)
    )
    labels = "".join(f"<span>{label}</span>" for label in data["labels"])
    spoken = ". ".join(f"{label} hours: {count} patients" for label, count in zip(data["labels"], data["counts"]))
    return (
        f'<section class="tile wide12"><h3>Hours of warning before onset</h3><div class="charttitle display">{data["title"]}</div>'
        f'<div class="bars" role="img" aria-label="{spoken}">{bars}</div><div class="barlabels" aria-hidden="true">{labels}</div>'
        f'<div class="barkey"><span><i style="background:{CHALK}"></i>inside the 12 hours before onset (timely)</span>'
        '<span><i style="background:#56616F"></i>more than 12 hours early</span><span>The number on each bar is patients. '
        "The labels are hours of warning.</span></div>"
        f'<p class="caption">{data["source"]}</p></section>'
    )


def curve_chart(data, narrow=False):
    """Utility at hospital B against the number of local patients added to training."""
    width, left, right = (380, 40, 14) if narrow else (1100, 52, 24)
    top, tall = 18, 220
    plot, bottom = width - left - right, top + tall
    points = [(pt["local"], pt["utility"]) for pt in data["points"]]       # (local patients added, utility)
    alone_local, alone_utility = data["alone"]["local"], data["alone"]["utility"]
    most = max(local for local, _ in points)
    values = [utility for _, utility in points] + [alone_utility]
    low, high = math.floor(min(values) * 20 - 0.4) / 20, math.ceil(max(values) * 20 + 0.2) / 20   # round out to steps of 0.05
    x = lambda v: left + v / most * plot
    y = lambda v: top + (high - v) / (high - low) * tall
    parts = [f'<rect x="{left}" y="{top}" width="{plot}" height="{tall}" rx="8" class="lane"/>']
    for i in range(round((high - low) * 20) + 1):
        level = low + i / 20
        parts.append(
            f'<line x1="{left}" x2="{left + plot}" y1="{y(level):.1f}" y2="{y(level):.1f}" class="grid"/>'
            f'<text x="{left - 6}" y="{y(level) + 4:.1f}" class="tick" text-anchor="end">{level:.2f}</text>'
        )
    line = " ".join(f"{x(local):.1f},{y(utility):.1f}" for local, utility in points)
    parts.append(f'<polyline points="{line}" class="scoreline"/>')
    for i, (local, utility) in enumerate(points):
        at, up = x(local), y(utility)
        named = not narrow or local in (0, 2000, 5000, most)      # on a phone the first labels would sit on each other
        last = i == len(points) - 1
        parts.append(f'<circle cx="{at:.1f}" cy="{up:.1f}" r="4.6" class="nowdot"><title>{local:,} local patients: utility {utility:.3f}</title></circle>')
        if named:
            side = "end" if last else "middle"
            parts.append(f'<text x="{at:.1f}" y="{bottom + 17}" class="tick" text-anchor="{side}">{local:,}</text>')
        if named or local == 1000:
            anchor, shift, drop = ("end", -2, 22) if last else ("start", 8, 16) if i < 3 else ("middle", 0, -12)
            parts.append(f'<text x="{at + shift:.1f}" y="{up + drop:.1f}" class="value halo" text-anchor="{anchor}">{utility:.3f}</text>')
    at, up = x(alone_local), y(alone_utility)
    words = "local patients only" if narrow else "local patients only, no hospital A"
    parts.append(f'<circle cx="{at:.1f}" cy="{up:.1f}" r="5" class="alonedot"/>')
    parts.append(f'<text x="{at - 10:.1f}" y="{up - 8:.1f}" class="tick halo" text-anchor="end">{words}: {alone_utility:.3f}</text>')
    x_title, y_title = data["x_title"], data["y_title"]
    parts.append(f'<text x="{left + plot}" y="{bottom + 36}" class="tick" text-anchor="end">{x_title}</text>')
    spoken = ". ".join(f"{local:,} local patients: {utility:.3f}" for local, utility in points)
    label = f"{y_title}. {spoken}. {alone_local:,} local patients alone: {alone_utility:.3f}."
    return f'<svg class="{"narrow" if narrow else "wide"}" viewBox="0 0 {width} {bottom + 44}" role="img" aria-label="{label}">{"".join(parts)}</svg>'


def curve_tile():
    data = SCOREBOARD.get("local_curve")
    if not data:
        return ""
    return (
        '<section class="tile wide12 after"><h3><em>After-the-fact</em>How much local data does a new hospital need?</h3>'
        f'<div class="charttitle display">{data["title"]}</div><p class="caption ytitle">{data["y_title"]}</p>'
        f'{curve_chart(data)}{curve_chart(data, narrow=True)}<p class="caption">{data["caption"]}</p></section>'
    )


def travel_tab():
    tiles = [
        f'<section class="tile wide12"><div class="headline display">{SCOREBOARD["headline"]}</div><p class="finding">{SCOREBOARD["finding"]}</p></section>',
        f'<section class="tile wide12"><h3>The full model at both hospitals</h3>{html_table(SCOREBOARD["two_hospitals"])}</section>',
        f'<section class="tile wide12"><h3>All planned models and both comparators</h3>{html_table(SCOREBOARD["models"], many=True)}</section>',
    ]
    if SCOREBOARD.get("travel_attempts"):
        tiles.append(
            f'<section class="tile wide12"><h3>Attempts to make it travel better</h3>{html_table(SCOREBOARD["travel_attempts"], many=True)}</section>'
        )
    tiles.append(warning_chart())
    tiles.append(curve_tile())
    tiles.append(f'<section class="tile wide12"><h3>Prediction against result</h3><div class="box">{SCOREBOARD["prediction_box"]}</div></section>')
    return f'<div class="grid12">{"".join(tiles)}</div>'


# ---------------------------------------------------------------- the page
START = CASES["cases"][0]["pid"]            # the page opens on the first saved patient, at its first hour
START_FIRST, START_LAST = hours_of(START)

with gr.Blocks(css=CSS, js=FORCE_DARK, theme=gr.themes.Base(), analytics_enabled=False, title="Sepsis Early Warning: does it travel?") as demo:
    gr.HTML(
        '<div class="topline"><span class="name display">Sepsis Early Warning</span>'
        '<span class="tag">Research prototype. Not medical advice.</span></div>'
    )
    if CASES["source"] != "B":
        gr.HTML('<div class="preview">Development preview: these patients and numbers are from hospital A dev, not hospital B.</div>')

    with gr.Tabs():
        with gr.Tab("Replay"):
            current = gr.State(START)
            with gr.Column(elem_classes="transport"):
                with gr.Row(elem_classes="buttons"):
                    pills = [
                        gr.Button(
                            case["label"], variant="primary" if case["pid"] == START else "secondary",
                            elem_classes=["pill", PILL_SHAPE.get(case["outcome"], "alert")], scale=0, min_width=0,
                        )
                        for case in CASES["cases"]
                    ]
                    gr.HTML("", elem_classes="spacer")
                    play = gr.Button("Play", variant="primary", elem_classes="go", scale=0, min_width=0)
                    step = gr.Button("Step one hour", elem_classes="go", scale=0, min_width=0)
                hour = gr.Slider(START_FIRST, START_LAST, value=START_FIRST, step=1, label="Hour in the unit", interactive=True, elem_classes="hour")
                rules = "; ".join(f'"{c["label"]}" is {c["rule"]}' for c in CASES["cases"])
                gr.HTML(f'<p class="caption">Four saved patients from {SITE}. Chosen by rule, not by eye: {rules}.</p>')
            timer = gr.Timer(0.7, active=False)

            board = gr.HTML(show(START, START_FIRST))

            def choose(pid):
                first, last = hours_of(pid)
                chosen = [gr.Button(variant="primary" if case["pid"] == pid else "secondary") for case in CASES["cases"]]
                return [pid, gr.Slider(minimum=first, maximum=last, value=first), show(pid, first), gr.Timer(active=False), "Play", *chosen]

            for button, case in zip(pills, CASES["cases"]):
                button.click(
                    lambda pid=case["pid"]: choose(pid), None,
                    [current, hour, board, timer, play, *pills], show_progress="hidden",
                )

            hour.change(show, [current, hour], board, show_progress="hidden")

            def advance(pid, now):
                if not pid:
                    return now, gr.Timer(active=False), "Play"
                first, last = hours_of(pid)
                nxt = min(int(now) + 1, last)
                if nxt >= last:
                    return nxt, gr.Timer(active=False), "Play"
                return nxt, gr.Timer(), gr.Button()

            step.click(lambda pid, now: advance(pid, now)[0], [current, hour], hour, show_progress="hidden")
            timer.tick(advance, [current, hour], [hour, timer, play], show_progress="hidden")

            def toggle(pid, label, now):
                if not pid:
                    return gr.Timer(active=False), "Play", now
                if label == "Play":
                    first, last = hours_of(pid)
                    restart = first if int(now) >= last else now   # at the end, Play starts over
                    return gr.Timer(active=True), "Pause", restart
                return gr.Timer(active=False), "Play", now

            play.click(toggle, [current, play, hour], [timer, play, hour], show_progress="hidden")

        with gr.Tab("Does it travel?"):
            gr.HTML(travel_tab())

        with gr.Tab("Limits and model card"):
            with gr.Column(elem_classes="limits"):
                gr.Markdown(LIMITS_TEXT)

if __name__ == "__main__":
    demo.launch(show_api=False)
