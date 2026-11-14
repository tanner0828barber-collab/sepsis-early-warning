# Sepsis Early Warning: does it travel?

**This project is a prototype and is not medical advice.**

An entry for the Veritas AI International High School Hackathon, Health track, built solo by Tanner Barber in the 48 hour window that opened on Sat 11/14/2026.

## The question

Can a sepsis early warning model trained at one hospital travel to another? We trained on public intensive care data from hospital A, froze the model and its alert threshold, and scored hospital B exactly once.

**The user:** an intensive care nurse deciding which patient to look at first.

## What happened

The full model, with its alert line set on hospital A so that 15% of patients who never develop sepsis get an alert:

| | Hospital A dev (4,068 patients) | Hospital B (20,000 patients) |
|---|---|---|
| Official challenge utility score | 0.368 | 0.238 (95% range 0.214 to 0.262) |
| Sepsis patients alerted at or before onset | 222 of 358 (62%) | 486 of 1,142 (43%) |
| of those, inside the 12 hours before onset | 60 | 109 |
| Sepsis patients missed | 136 | 656 |
| False alarms | 556 of 3,710 (15.0%) | 1,358 of 18,858 (7.2%) |

- **It travels only partly.** Utility fell by about a third.
- **It got quieter, not noisier.** Hospital B measures lab values less often and its blood pressures read higher, so the model scored everyone lower there.
- **Time in the unit explains a lot.** A model that sees only hours in the unit scored 0.242 at hospital B, about the same as the full model.
- **Most alerts are very early.** We count an alert more than 12 hours before onset separately and never call it a catch.
- **The bedside rule (modified SIRS)** alerted by onset on 966 of 1,142 sepsis patients at hospital B, but raised a false alarm on 67% of everyone else.

Hospital B is a site-held-out evaluation set, not a blinded test. The full disclosure is in [DEFINITIONS.md](DEFINITIONS.md).

## Where to look

- **[sepsis_early_warning.ipynb](sepsis_early_warning.ipynb)**: the whole project top to bottom, with outputs saved. It retrains the models and reproduces the frozen numbers.
- [DECISIONS.md](DECISIONS.md): every choice with its time and reason, the prediction committed before any model was trained, and how it compared with the result.
- [DEFINITIONS.md](DEFINITIONS.md): what "caught," "very early," "missed," "false alarm," and "onset" mean here.
- [SPEC.md](SPEC.md): the one page plan written before any code.
- `src/` holds the working code, `tests/` the checks (including that no input can see the future), `results/` every reported number, and `app/` the replay tool.

## Run it

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests          # the checks
.venv/bin/jupyter notebook sepsis_early_warning.ipynb
.venv/bin/python app/app.py                  # the replay tool, at http://localhost:7860
```

The data is not stored in this repository. The notebook downloads it from the public source.

## Data and credits

- **Data:** PhysioNet / Computing in Cardiology Challenge 2019, training sets A and B. Public, de-identified, license CC BY 4.0. https://physionet.org/content/challenge-2019/1.0.0/
- **Challenge paper:** Reyna MA, Josef CS, Jeter R, et al. "Early Prediction of Sepsis from Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019." Critical Care Medicine, 2020.
- **Scoring:** the challenge organizers' own scoring script, unchanged, BSD 2-Clause license, in `third_party/`.
- **Libraries:** pandas, NumPy, scikit-learn, Matplotlib, Plotly, Gradio, Jupyter.
- **AI tools:** Claude Code wrote most of the code from Tanner's decisions and ran the literature checks. ChatGPT (Codex) was asked for an outside review. Tanner chose the question, the user, the alert budget and the definitions, adopted the prediction before any model was trained, and decided what to keep.
