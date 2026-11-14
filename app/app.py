"""Sepsis Early Warning: the replay tool.

Run it:  .venv/bin/python app/app.py

What it does: loads the frozen model, takes one saved, de-identified patient,
and replays the stay hour by hour. At each hour the model is given only the
rows up to that hour and really scores them. Nothing is typed in by the user
and nothing is stored.

This project is a prototype and is not medical advice.
"""
import json
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import build_features  # noqa: E402

APP_DATA = Path(__file__).resolve().parent / "data"

# ---------------------------------------------------------------- frozen things
FROZEN = json.loads((ROOT / "results" / "frozen.json").read_text())
THRESHOLD = FROZEN["models"]["M3"]["threshold"]
COLUMNS = FROZEN["models"]["M3"]["columns"]
MODEL = joblib.load(ROOT / "models" / "M3.joblib")

CASES = json.loads((APP_DATA / "cases.json").read_text())
PATIENTS = pd.read_csv(APP_DATA / "patients.csv")
PATIENTS["hosp"] = "saved"
SCOREBOARD = json.loads((APP_DATA / "scoreboard.json").read_text())
LIMITS_TEXT = (Path(__file__).resolve().parent / "limits.md").read_text()

# ---------------------------------------------------------------- look and feel
PAGE, TEXT, SECONDARY, ACCENT = "#FAFAF7", "#111418", "#5B6470", "#2450D8"
ALERT_TINT, ALERT_LABEL = "#FBDADA", "#A32222"
WATCH_TINT, WATCH_LABEL = "#FBEFC9", "#7A5600"
CLEAR_TINT, CLEAR_LABEL = "#DDF3E6", "#14713F"

# Names and units, as listed in the dataset's own description on PhysioNet.
VITAL_STRIPS = [
    ("HR", "Heart rate", "beats per minute", (40, 160)),
    ("Temp", "Temperature", "°C", (34, 41)),
    ("Resp", "Breathing rate", "breaths per minute", (5, 45)),
    ("MAP", "Mean arterial pressure", "mm Hg", (40, 130)),
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
STALE_AFTER = {"vital": 4, "lab": 24}  # hours; our own choice, not a clinical standard

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
.gradio-container, .gradio-container * {{ font-family: 'Inter', system-ui, sans-serif !important; }}
.gradio-container {{ background: {PAGE} !important; color: {TEXT}; max-width: 1000px !important; margin: 0 auto !important; }}
.narrow {{ max-width: 760px; }}
.topline {{ display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }}
.topline .name {{ font-size: 18px; font-weight: 600; color: {TEXT}; }}
.topline .tag {{ font-size: 14px; color: {SECONDARY}; border: 1px solid #D5D8DD; border-radius: 999px; padding: 4px 12px; }}
.promise {{ font-size: 24px; font-weight: 600; line-height: 1.25; color: {TEXT}; margin: 16px 0 8px; max-width: 760px; }}
.proof {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 16px 0; }}
.proof .cell {{ background: #FFFFFF; border: 1px solid #E4E6EA; border-radius: 12px; padding: 16px; }}
.proof .big {{ font-size: 40px; font-weight: 600; line-height: 1.05; letter-spacing: -0.02em; color: {TEXT}; font-variant-numeric: tabular-nums; }}
.proof .what {{ font-size: 16px; color: {TEXT}; margin-top: 8px; line-height: 1.4; }}
.proof .where {{ font-size: 14px; color: {SECONDARY}; margin-top: 4px; line-height: 1.4; }}
.caption {{ font-size: 14px; color: {SECONDARY}; line-height: 1.5; }}
.card {{ border-radius: 12px; padding: 16px; margin: 8px 0; min-height: 96px; }}
.card .word {{ font-size: 24px; font-weight: 600; line-height: 1.25; }}
.card .more {{ font-size: 16px; color: {TEXT}; margin-top: 8px; line-height: 1.5; }}
.card.clear {{ background: {CLEAR_TINT}; }} .card.clear .word {{ color: {CLEAR_LABEL}; }}
.card.alert {{ background: {ALERT_TINT}; }} .card.alert .word {{ color: {ALERT_LABEL}; }}
.card.watch {{ background: {WATCH_TINT}; }} .card.watch .word {{ color: {WATCH_LABEL}; }}
.card.empty {{ background: #FFFFFF; border: 1px dashed #C9CDD3; }} .card.empty .word {{ color: {SECONDARY}; font-size: 18px; font-weight: 500; }}
table.plain {{ border-collapse: collapse; width: 100%; font-size: 16px; }}
table.plain th {{ text-align: left; font-weight: 600; color: {SECONDARY}; font-size: 14px; padding: 8px; border-bottom: 1px solid #D5D8DD; }}
table.plain td {{ padding: 8px; border-bottom: 1px solid #ECEEF1; color: {TEXT}; font-variant-numeric: tabular-nums; vertical-align: top; }}
table.plain td.num, table.plain th.num {{ text-align: right; }}
.range {{ color: {SECONDARY}; font-size: 14px; }}
.chip {{ display: inline-block; font-size: 14px; border-radius: 999px; padding: 0 8px; margin-left: 8px; }}
.chip.stale {{ background: {WATCH_TINT}; color: {WATCH_LABEL}; }}
.chip.none {{ background: #ECEEF1; color: {SECONDARY}; }}
.finding {{ font-size: 18px; line-height: 1.5; color: {TEXT}; max-width: 760px; margin: 8px 0 16px; }}
.box {{ background: #FFFFFF; border: 1px solid #E4E6EA; border-radius: 12px; padding: 16px; font-size: 16px; line-height: 1.5; max-width: 760px; }}
.preview {{ background: {WATCH_TINT}; color: {WATCH_LABEL}; border-radius: 8px; padding: 8px 12px; font-size: 16px; font-weight: 500; }}
button.pill {{ border-radius: 999px !important; min-height: 48px !important; font-size: 16px !important; }}
button.go {{ min-height: 48px !important; font-size: 16px !important; }}
footer {{ display: none !important; }}
@media (prefers-reduced-motion: reduce) {{ * {{ transition: none !important; animation: none !important; }} }}
"""

FORCE_LIGHT = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'light') {
    url.searchParams.set('__theme', 'light');
    window.location.replace(url.href);
  }
}
"""


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
    inputs["score"] = MODEL.predict_proba(inputs[COLUMNS])[:, 1]
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


# ---------------------------------------------------------------- pieces of the screen
def plural(n, word):
    n = int(n)
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def make_chart(pid, hour):
    first, last = hours_of(pid)
    seen = seen_so_far(pid, hour)
    rows = record(pid)
    rows = rows[rows.ICULOS <= hour]
    facts = truth(pid)

    titles = ["Relative risk score (1 = the alert line)"] + [
        f"{name} ({unit})" for _, name, unit, _ in VITAL_STRIPS
    ]
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.06,
        row_heights=[0.36, 0.16, 0.16, 0.16, 0.16], subplot_titles=titles,
    )

    # the score, drawn as a multiple of the frozen alert line
    relative = np.clip(seen["score"] / THRESHOLD, 0.03, 30)
    fig.add_trace(go.Scatter(
        x=seen.ICULOS, y=relative, mode="lines+markers", line=dict(color=ACCENT, width=2),
        marker=dict(size=5), name="Relative risk score",
        hovertemplate="Hour %{x}: %{y:.2f} times the alert line<extra></extra>",
    ), row=1, col=1)
    fig.add_hline(y=1, line=dict(color=ALERT_LABEL, width=1.5, dash="dash"), row=1, col=1)
    fig.add_annotation(
        x=first, y=np.log10(1), xref="x", yref="y", text="Alert line, frozen on hospital A",
        showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=13, color=ALERT_LABEL),
    )
    fig.update_yaxes(
        type="log", range=[np.log10(0.03), np.log10(30)], tickvals=[0.1, 1, 10],
        ticktext=["0.1", "1", "10"], row=1, col=1,
    )

    # markers, each a line plus a word, shown only once the replay reaches them
    alerting = seen[seen["score"] >= THRESHOLD]
    marks = []
    if not alerting.empty:
        marks.append((int(alerting.ICULOS.iloc[0]), "First alert", ALERT_LABEL, 1.0))
    if facts["sepsis"]:
        if facts["label_start"] <= hour:
            marks.append((facts["label_start"], "Label start", SECONDARY, 0.86))
        if facts["onset"] <= hour:
            marks.append((facts["onset"], "Onset", TEXT, 0.72))
    for x, word, color, height in marks:
        fig.add_vline(x=x, line=dict(color=color, width=1.5, dash="dot"))
        fig.add_annotation(
            x=x, y=height, xref="x", yref="paper", text=word, showarrow=False,
            xanchor="left", xshift=4, font=dict(size=13, color=color),
            bgcolor="rgba(250,250,247,0.9)",
        )

    # vital sign strips: a dot is a real measurement, a faint dashed line is a carried value
    for i, (column, name, unit, limits) in enumerate(VITAL_STRIPS, start=2):
        carried = seen[f"last_{column}"].clip(*limits)
        fig.add_trace(go.Scatter(
            x=seen.ICULOS, y=carried, mode="lines", line=dict(color="#9AA1AC", width=1, dash="dash", shape="hv"),
            hoverinfo="skip",
        ), row=i, col=1)
        measured = rows[rows[column].notna()]
        fig.add_trace(go.Scatter(
            x=measured.ICULOS, y=measured[column].clip(*limits), mode="markers",
            marker=dict(size=6, color=TEXT), name=name,
            hovertemplate="Hour %{x}: %{y:.1f} " + unit + " (measured)<extra></extra>",
        ), row=i, col=1)
        fig.update_yaxes(range=list(limits), nticks=3, row=i, col=1)

    fig.update_xaxes(range=[first - 0.5, last + 0.5], showgrid=False)
    fig.update_xaxes(title_text="Hours in the unit", row=5, col=1)
    fig.update_yaxes(gridcolor="#ECEEF1", zeroline=False)
    fig.update_annotations(font_family="Inter, system-ui, sans-serif")
    for annotation in fig.layout.annotations[:5]:   # the strip titles: left aligned, small
        annotation.update(x=0, xanchor="left", font=dict(size=14, color=SECONDARY))
    fig.update_layout(
        height=720, showlegend=False, paper_bgcolor=PAGE, plot_bgcolor="#FFFFFF",
        margin=dict(l=48, r=16, t=40, b=48),
        font=dict(family="Inter, system-ui, sans-serif", size=14, color=TEXT),
        transition_duration=0,
    )
    return fig


def status_card(pid, hour):
    first, last = hours_of(pid)
    seen = seen_so_far(pid, hour)
    facts = truth(pid)
    alerting = seen[seen["score"] >= THRESHOLD]
    now_on = bool(seen["score"].iloc[-1] >= THRESHOLD)

    if alerting.empty:
        kind, word = "clear", "✓ No alert"
        more = [f"Hour {hour}. The score has stayed under the alert line."]
    else:
        fired = int(alerting.ICULOS.iloc[0])
        kind, word = "alert", "▲ ALERT: look at this patient first"
        more = [f"First fired at hour {fired}."]
        if not now_on:
            more.append("The score is back under the alert line now.")

    # what really happened, revealed only when the replay reaches it
    if facts["sepsis"] and hour >= facts["onset"]:
        by_onset = alerting[alerting.ICULOS <= facts["onset"]]
        if by_onset.empty:
            more.append(f"Sepsis onset was at hour {facts['onset']}. No alert before onset. This is a miss.")
        else:
            warned = facts["onset"] - int(by_onset.ICULOS.iloc[0])
            more.append(f"Sepsis onset was at hour {facts['onset']}. The alert came {plural(warned, 'hour')} before.")
    elif facts["sepsis"] and hour >= last:
        more.append("The record ends before the onset hour.")
    if not facts["sepsis"] and hour >= last:
        if alerting.empty:
            more.append("End of the record. This patient never developed sepsis, and there was no alert.")
        else:
            more.append("End of the record. This patient never developed sepsis. This alert was a false alarm.")

    return f'<div class="card {kind}" role="status"><div class="word">{word}</div><div class="more">{" ".join(more)}</div></div>'


def seen_table(pid, hour):
    now = seen_so_far(pid, hour).iloc[-1]
    lines = []
    for column, name, unit, kind in SEEN_TABLE:
        value, age = now[f"last_{column}"], now[f"hours_since_{column}"]
        if pd.isna(value):
            lines.append(f'<tr><td>{name}</td><td class="num"><span class="chip none">not measured</span></td><td></td></tr>')
            continue
        when = "this hour" if age == 0 else f"{plural(age, 'hour')} ago"
        stale = '<span class="chip stale">stale</span>' if age > STALE_AFTER[kind] else ""
        lines.append(f'<tr><td>{name}</td><td class="num">{value:g} {unit}</td><td>measured {when}{stale}</td></tr>')
    return (
        '<table class="plain"><thead><tr><th>Input</th><th class="num">Latest value</th><th>How old</th></tr></thead>'
        f'<tbody>{"".join(lines)}</tbody></table>'
        '<p class="caption">"Stale" means a vital sign older than 4 hours or a lab older than 24 hours. '
        "That cutoff is our own choice for this display. Units are as listed in the dataset's description. "
        "The model also sees trends, time in the unit, and how long ago each value was measured.</p>"
    )


def show(pid, hour):
    """Everything on the Replay tab for one patient at one hour."""
    if not pid:
        empty = '<div class="card empty"><div class="word">Choose a saved patient above to start the replay.</div></div>'
        return None, empty, ""
    first, last = hours_of(pid)
    hour = int(min(max(hour, first), last))
    return make_chart(pid, hour), status_card(pid, hour), seen_table(pid, hour)


# ---------------------------------------------------------------- static blocks
def proof_strip():
    s = SCOREBOARD["proof"]
    cells = "".join(
        f'<div class="cell"><div class="big">{c["big"]}</div><div class="what">{c["what"]}</div>'
        f'<div class="where">{c["where"]}</div></div>'
        for c in s
    )
    return f'<div class="proof">{cells}</div>'


def html_table(table):
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
    return f'<table class="plain"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>{caption}'


def warning_chart():
    data = SCOREBOARD.get("warning_chart")
    if not data:
        return None
    fig = go.Figure(go.Bar(
        x=data["labels"], y=data["counts"], text=data["counts"], textposition="outside",
        marker_color=ACCENT, hovertemplate="%{x}: %{y} patients<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=data["title"], font=dict(size=18)), height=380, paper_bgcolor=PAGE,
        plot_bgcolor="#FFFFFF", margin=dict(l=48, r=16, t=64, b=64),
        font=dict(family="Inter, system-ui, sans-serif", size=14, color=TEXT),
        xaxis_title="Hours of warning before onset", yaxis_title="Patients",
    )
    fig.update_yaxes(gridcolor="#ECEEF1")
    return fig


# ---------------------------------------------------------------- the page
with gr.Blocks(css=CSS, js=FORCE_LIGHT, analytics_enabled=False, title="Sepsis Early Warning: does it travel?") as demo:
    gr.HTML(
        '<div class="topline"><span class="name">Sepsis Early Warning</span>'
        '<span class="tag">Research prototype. Not medical advice.</span></div>'
    )
    if CASES["source"] != "B":
        gr.HTML('<div class="preview">Development preview: these patients and numbers are from hospital A dev, not hospital B.</div>')

    with gr.Tabs():
        with gr.Tab("Replay"):
            gr.HTML(
                '<div class="promise">An early warning for sepsis, trained at one hospital and tested at another. '
                "Replay a real, de-identified patient hour by hour and see when it would have alerted.</div>"
            )
            gr.HTML(proof_strip())

            current = gr.State("")
            gr.HTML('<div class="caption" style="font-size:16px;color:#111418;font-weight:500;">Saved patients</div>')
            with gr.Row():
                pills = [gr.Button(case["label"], elem_classes="pill") for case in CASES["cases"]]
            rules = "; ".join(f'"{c["label"]}" is {c["rule"]}' for c in CASES["cases"])
            gr.HTML(f'<p class="caption">Saved from {CASES["site"]}. Chosen by rule, not by eye: {rules}.</p>')

            with gr.Row():
                hour = gr.Slider(1, 48, value=1, step=1, label="Hour in the unit", scale=4, interactive=True)
                play = gr.Button("Play", variant="primary", elem_classes="go", scale=1)
                step = gr.Button("Step one hour", elem_classes="go", scale=1)
            timer = gr.Timer(0.7, active=False)

            status = gr.HTML(show("", 1)[1])
            chart = gr.Plot(show_label=False)
            gr.HTML('<div class="caption" style="font-size:16px;color:#111418;font-weight:500;margin-top:16px;">What the model could see at this hour</div>')
            seen_html = gr.HTML("")
            gr.HTML(
                '<p class="caption">Retrospective replay of public, de-identified records. Nothing you do here is stored. '
                "This project is a prototype and is not medical advice.</p>"
            )

            def choose(pid):
                first, last = hours_of(pid)
                fig, card, table = show(pid, first)
                return pid, gr.Slider(minimum=first, maximum=last, value=first), fig, card, table, gr.Timer(active=False), "Play"

            for button, case in zip(pills, CASES["cases"]):
                button.click(
                    lambda pid=case["pid"]: choose(pid), None,
                    [current, hour, chart, status, seen_html, timer, play],
                )

            hour.change(show, [current, hour], [chart, status, seen_html], show_progress="hidden")

            def advance(pid, now):
                if not pid:
                    return now, gr.Timer(active=False), "Play"
                first, last = hours_of(pid)
                nxt = min(int(now) + 1, last)
                if nxt >= last:
                    return nxt, gr.Timer(active=False), "Play"
                return nxt, gr.Timer(), gr.Button()

            step.click(lambda pid, now: advance(pid, now)[0], [current, hour], hour)
            timer.tick(advance, [current, hour], [hour, timer, play], show_progress="hidden")

            def toggle(pid, label, now):
                if not pid:
                    return gr.Timer(active=False), "Play", now
                if label == "Play":
                    first, last = hours_of(pid)
                    restart = first if int(now) >= last else now   # at the end, Play starts over
                    return gr.Timer(active=True), "Pause", restart
                return gr.Timer(active=False), "Play", now

            play.click(toggle, [current, play, hour], [timer, play, hour])

        with gr.Tab("Does it travel?"):
            gr.HTML(f'<div class="promise">{SCOREBOARD["headline"]}</div>')
            gr.HTML(f'<div class="finding">{SCOREBOARD["finding"]}</div>')
            gr.HTML(html_table(SCOREBOARD["two_hospitals"]))
            gr.HTML('<div class="promise" style="font-size:18px;">All planned models and both comparators</div>')
            gr.HTML(html_table(SCOREBOARD["models"]))
            if SCOREBOARD.get("travel_attempts"):
                gr.HTML('<div class="promise" style="font-size:18px;">Attempts to make it travel better</div>')
                gr.HTML(html_table(SCOREBOARD["travel_attempts"]))
            figure = warning_chart()
            if figure is not None:
                gr.Plot(figure, show_label=False)
                gr.HTML(f'<p class="caption">{SCOREBOARD["warning_chart"]["source"]}</p>')
            gr.HTML(f'<div class="box">{SCOREBOARD["prediction_box"]}</div>')

        with gr.Tab("Limits and model card"):
            gr.Markdown(LIMITS_TEXT, elem_classes="narrow")

if __name__ == "__main__":
    demo.launch(show_api=False)
