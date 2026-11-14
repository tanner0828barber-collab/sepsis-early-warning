# Sepsis Early Warning: moved to a second hospital, it gave a timely alert to only 109 of 1,142 sepsis patients

**Track:** Health. **Team:** Tanner Barber, solo. This project is a prototype and is not medical advice.

## The problem
About 1.7 million adults in the U.S. develop sepsis each year, and at least 350,000 die in the hospital or are discharged to hospice ([CDC](https://www.cdc.gov/sepsis/about/)). An intensive care nurse cannot study every chart every hour. Warning models exist, but each is built at one hospital. Does one still work when it moves?

## What we built
I built a model that reads an intensive care patient's record hour by hour (7 vital signs, 26 labs, age, sex, hours in the unit) and alerts when its score crosses a fixed line. **The user is an intensive care nurse deciding which patient to look at first.** A replay tool plays back a real, de-identified stay and shows what the model saw each hour and when it alerted.
Try it: `app/app.py`, steps in the [README](https://github.com/tanner0828barber-collab/sepsis-early-warning#run-it). Notebook: [sepsis_early_warning.ipynb](https://github.com/tanner0828barber-collab/sepsis-early-warning/blob/main/sepsis_early_warning.ipynb). Code: https://github.com/tanner0828barber-collab/sepsis-early-warning

## What we added
Our own idea. My user is an intensive care nurse, my data is public records from two hospitals, and my question is whether a model trained at hospital A still works at hospital B.

## How it works
**Data:** [PhysioNet / Computing in Cardiology Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/), CC BY 4.0, 40,336 patients, one row per hour. I dropped one column that is 100% empty at hospital A. One thing I found by reading rows: records stop a few hours after sepsis onset.
**Split:** hospital A by patient: 13,827 fit, 2,441 tune, 4,068 dev. I set the alert line once on A dev so that at most 15% of patients who never develop sepsis get an alert (so A dev numbers are calibration numbers). I then froze everything and scored hospital B (20,000 patients) exactly once.
**Disclosure:** hospital B is site-held-out, not blind. Its counts, age and sex mix, and sepsis rates were summarized before modeling, but no model, input, or alert line ever used it.
**Model:** never alert (utility 0 by definition), a modified SIRS bedside rule, a model that sees only hours in the unit, then gradient boosting on 100 inputs. Each step shows what the next one adds.

## Results
At hospital B, of 1,142 sepsis patients, **109 (9.5%) got their first alert in the 12 hours before onset**, 377 were alerted earlier than that (very early), and **656 (57%) were missed**. Of 18,858 patients who never developed sepsis, **1,358 (7.2%) got a false alarm**. That is 12.5 false alarms per timely catch, and 115 alerts per 1,000 patient-days.

| Method | Utility, A dev | Utility, B (95% range) | Timely at B | Missed at B | False alarms at B |
|---|---|---|---|---|---|
| Modified SIRS rule | 0.090 | -0.014 (-0.051 to 0.023) | 306 | 176 | 12,630 (67.0%) |
| Hours in the unit only | 0.312 | 0.242 (0.217 to 0.268) | 79 | 717 | 2,581 (13.7%) |
| Full model | 0.368 | 0.238 (0.214 to 0.262) | 109 | 656 | 1,358 (7.2%) |

Utility is the challenge's official score (1 is perfect). **At hospital B the full model did not beat the model that only knows the hour.**

**The test nobody asked for.** Before any model was trained I predicted: utility falls by a third to a half (right: down 35%), false alarms go above 15% (wrong: 7.2%), and the full model loses more than a body-signals-only model (mostly wrong). The model got quieter, not noisier: hospital B records 0.94 lab values per patient-hour against 1.64 at A and its blood pressures read higher, so everyone scored lower.

**Where it fails** (picked by rule, by patient ID; all five are in the notebook):
1. Missed, man aged 50: vital signs normal throughout, and no lab measured until the onset hour.
2. Missed, man aged 77: fever for hours and white cells at 22.4. The SIRS rule would have fired. The model's score peaked at 0.34 of the alert line.
3. False alarm, man aged 36: the score jumped for one hour when a set of labs appeared and no vital signs were recorded.

Pattern: first-day sepsis. When the label starts before hour 24, 57 of 514 were alerted by onset (11%), against 429 of 628 later (68%). Late-entry cases, labeled from their first row: 14 of 223 caught (SIRS rule: 150 of 223).

**Who it works worse for** (hospital B, alerted by onset): women 210 of 495 (42%), men 276 of 647 (43%). Under 50: 116 of 276 (42%). 50 to 64: 141 of 351 (40%). 65 to 79: 183 of 386 (47%). **80 and over: 46 of 129 (36%), the worst group.** False alarms: women 6.2%, men 8.1%. The data has no race or ethnicity field, so those were not checked.

**Limits:** retrospective, and never tried by a nurse. The label is built partly from when staff ordered antibiotics and cultures, so the model partly learns hospital habits.

## What we'd do next
I would set the alert line for each hospital from its own unlabeled data (a planned first try reached 629 of 1,142 alerted by onset at 17.1% false alarms), then test at a hospital nobody has looked at.

## Sources
Data: PhysioNet Challenge 2019, CC BY 4.0 (Reyna et al., Critical Care Medicine, 2020). Scoring: the organizers' script, unchanged (BSD 2-Clause). No pretrained models. Libraries: pandas, NumPy, scikit-learn, Plotly, Gradio, Jupyter. Evidence behind the prediction is cited in [DECISIONS.md](https://github.com/tanner0828barber-collab/sepsis-early-warning/blob/main/DECISIONS.md).

| AI tool | What it did | What I did |
|---|---|---|
| Claude Code | Wrote most of the code, drafted the prediction from published evidence | Chose the question, user, alert budget and definitions. Adopted the prediction before training. Decided what to keep and when to freeze |
| ChatGPT (Codex) | Outside review of the frozen project | Asked for it, read it, acted on its findings |
