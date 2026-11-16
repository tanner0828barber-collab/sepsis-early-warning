# Sepsis Early Warning

**Does a sepsis alarm built at one hospital still work at another?**

Tanner Barber, solo. Health track. This project is a prototype and is not medical advice.

| 109 of 1,142 | 656 of 1,142 | 7.2% | 0.368 to 0.238 |
|---|---|---|---|
| sepsis patients got a timely first alert at the second hospital | sepsis patients were missed there | of patients without sepsis got a false alarm (1,358 of 18,858) | official utility score, home hospital to second hospital |

## The problem
About 1.7 million adults in the U.S. develop sepsis each year, and at least 350,000 die in the hospital or are discharged to hospice ([CDC](https://www.cdc.gov/sepsis/about/)). A nurse cannot study every chart every hour. Warning models exist, but each one is built at a single hospital. Does it still work when it moves?

## What I built
I built a model that reads an intensive care patient's record hour by hour (7 vital signs, 26 labs, age, sex, hours in the unit) and alerts when its score crosses a fixed line. **The user is an intensive care nurse deciding which patient to look at first.** My tool replays real, de-identified stays hour by hour and ranks a 12 bed unit by score.

Try it: `app/app.py`, steps in the [README](https://github.com/tanner0828barber-collab/sepsis-early-warning#run-it). Notebook: [sepsis_early_warning.ipynb](https://github.com/tanner0828barber-collab/sepsis-early-warning/blob/main/sepsis_early_warning.ipynb). Code: [github.com/tanner0828barber-collab/sepsis-early-warning](https://github.com/tanner0828barber-collab/sepsis-early-warning)

## What I added
My own idea. My user is an intensive care nurse, my data is public records from two hospitals, and my question is whether a model trained at hospital A still works at hospital B.

## How it works
- **Data:** [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/), CC BY 4.0. 40,336 patients, one row per hour. I dropped one column that is 100% empty at hospital A. Reading raw rows, I found that records stop a few hours after sepsis onset.
- **Split:** hospital A by patient: 13,827 fit, 2,441 tune, 4,068 dev. I set the alert line once on A dev so that at most 15% of patients who never develop sepsis get an alert.
- **Test:** I froze everything, then scored hospital B (20,000 patients) exactly once.
- **Models:** never alert, a modified SIRS bedside rule, hours in the unit only, then gradient boosting on 100 inputs.
- **Disclosure:** hospital B is site-held-out, not blind. Its counts, age and sex mix, and sepsis rates were summarized before modeling. No model, input, or alert line ever used it.

## Results

| Method | Utility, A dev | Utility, B (95% range) | Timely at B | Missed at B | False alarms at B |
|---|---|---|---|---|---|
| Modified SIRS rule | 0.090 | -0.014 (-0.051 to 0.023) | 306 | 176 | 12,630 (67.0%) |
| Hours in the unit only | 0.312 | 0.242 (0.217 to 0.268) | 79 | 717 | 2,581 (13.7%) |
| **Full model** | **0.368** | **0.238** (0.214 to 0.262) | **109** | **656** | **1,358 (7.2%)** |

Utility is the challenge's official score (1 is perfect). "Timely" means the first alert came in the 12 hours before onset. **At hospital B my full model did not beat the model that only knows the hour.** Another 377 sepsis patients were alerted more than 12 hours early, which I never count as a catch.

**The test nobody asked for.** Before training any model I wrote down three predictions. Utility falls by a third to a half: right, down 35%. False alarms go above 15%: wrong, 7.2%. The full model loses more than a body-signals-only model: mostly wrong. The model got quieter, not noisier. Hospital B records 0.94 lab values per patient-hour against 1.64 at A, and its blood pressures read higher.

**Where it fails** (picked by rule, by patient ID):

- Missed, man aged 50: vital signs normal throughout, and no lab measured until the onset hour.
- Missed, man aged 77: fever for hours and white cells at 22.4. The SIRS rule would have fired.
- False alarm, man aged 36: the score jumped for one hour when labs appeared with no vital signs.

The pattern is first-day sepsis. When the label starts before hour 24, 57 of 514 were alerted by onset (11%), against 429 of 628 later (68%).

**Who it works worse for.** Patients aged 80 and over: 46 of 129 alerted by onset (36%), against 40% to 47% in the younger groups. Women 42%, men 43%. False alarms: women 6.2%, men 8.1%. The data has no race or ethnicity field.

**Limits.** Retrospective, and never tried by a nurse. The label depends partly on when staff ordered antibiotics and cultures, so the model partly learns hospital habits.

**After the frozen run** (after-the-fact, not a like for like improvement). With hospital B's labeled patients in training and a 25% false alarm cap, one model scored 0.402 at A and 0.406 at B. It has seen B, so it cannot claim to travel. Run backwards, a 15% budget set at B gave 45.0% false alarms at A.

## What I'd do next
I would set each hospital's alert line from its own unlabeled data (a first try reached 629 of 1,142 alerted by onset at 17.1% false alarms), then test at a hospital nobody has looked at.

## Sources
Data: PhysioNet Challenge 2019, CC BY 4.0 (Reyna et al., Critical Care Medicine, 2020). Scoring: the organizers' script, unchanged. No pretrained models. Every dataset, library, tutorial and AI tool is credited with its link in the [README](https://github.com/tanner0828barber-collab/sepsis-early-warning#data-and-credits). All results and every decision: [DECISIONS.md](https://github.com/tanner0828barber-collab/sepsis-early-warning/blob/main/DECISIONS.md).

| AI tool | What it did | What I did |
|---|---|---|
| [Claude Code](https://claude.com/claude-code) | Wrote most of the code and drafted the prediction from published evidence | Chose the question, user, alert budget and definitions. Adopted the prediction before training. Decided what to keep and when to freeze |
| [ChatGPT (Codex)](https://openai.com/codex/) | Outside review of the frozen project | Asked for it, read it, acted on its findings |
