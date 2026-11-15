# Sepsis Early Warning: does it travel?

**This project is a prototype and is not medical advice.**

An entry for the Veritas AI International High School Hackathon, Health track, built solo by Tanner Barber in the 48 hour window that opened on Sat 11/14/2026.

## The question

Can a sepsis early warning model trained at one hospital travel to another? I trained on public intensive care data from hospital A, froze the model and its alert threshold, and scored hospital B exactly once.

**The user:** an intensive care nurse deciding which patient to look at first.

## What happened

The full model, with its alert line set on hospital A so that 15% of patients who never develop sepsis get an alert:

| | Hospital A dev (4,068 patients, the calibration set) | Hospital B (20,000 patients) |
|---|---|---|
| Timely catches: first alert in the 12 hours before onset | 60 of 358 (17%) | **109 of 1,142 (9.5%)** |
| Alerted very early: more than 12 hours before onset | 162 of 358 | 377 of 1,142 |
| Sepsis patients missed | 136 of 358 | 656 of 1,142 (57%) |
| False alarms | 556 of 3,710 (15.0%) | 1,358 of 18,858 (7.2%) |
| False alarms per timely catch | 9.3 | 12.5 |
| Alerts per 1,000 patient-days | 232 | 115 |
| Official challenge utility score | 0.368 | 0.238 (95% range 0.214 to 0.262) |

The alert line was set on hospital A dev, so its 15.0% false alarm share is true by construction. "Alerted at or before onset" (222 of 358 at A dev, 486 of 1,142 at B) adds the timely and very early rows together.

- **Timely catches were rare.** At hospital B, 109 of 1,142 sepsis patients got their first alert in the 12 hours before onset.
- **It travels only partly.** Utility fell by about a third.
- **It got quieter, not noisier.** Hospital B measures lab values less often and its blood pressures read higher, so the model scored everyone lower there.
- **Time in the unit explains a lot.** A model that sees only hours in the unit scored 0.242 at hospital B, about the same as the full model.
- **Most alerts are very early.** I count an alert more than 12 hours before onset separately and never call it a catch.
- **The bedside rule (modified SIRS)** alerted by onset on 966 of 1,142 sepsis patients at hospital B, but raised a false alarm on 67% of everyone else.

Hospital B is a site-held-out evaluation set, not a blinded test. The full disclosure is in [DEFINITIONS.md](DEFINITIONS.md).

## After the hospital B run: version 2 (after-the-fact)

**Version 1 above stays the result of record.** After hospital B was scored I ran seven more experiments. All seven are after-the-fact: hospital B had already been seen, so none of them is a test of whether a model travels. Most use a 25% false alarm cap where version 1 used a 15% budget, so they are a different test and not a like for like improvement on 0.238. Each plan was written into the "Version 2" section of [DECISIONS.md](DECISIONS.md) before it ran, and every number is in `results/after_the_fact_v2_*.csv`.

| Experiment | Scored on | Utility | Sepsis patients missed | False alarms |
|---|---|---|---|---|
| One model trained on both hospitals, 25% cap at each hospital | Hospital A, 16,268 patients | 0.402 | 399 of 1,432 | 25.4% |
| The same both-hospitals model | Hospital B, 20,000 patients | 0.406 | 220 of 1,142 | 25.2% |
| Frozen version 1 model, hospital A's 25% alert line unchanged | Hospital B, 18,000 patients | 0.265 | 484 of 1,024 | 15.2% |
| Frozen version 1 model, hospital B's own alert line, no labels | The same 18,000 | 0.271 | 383 of 1,024 | 28.0% |
| Backwards: trained at hospital B, 15% budget set there | Hospital A, 20,336 patients | 0.330 | 422 of 1,790 | 45.0% |

- **With hospital B's labeled patients in training, the model reaches 0.406 there.** That is the best version 2 model. It was scored with nested cross-validation, which means no patient is scored by a model or an alert line that saw them. It has seen both hospitals, so it cannot claim to travel. It answers a different question: what do you get when a hospital has its own labeled data?
- **An alert line does not travel, in either direction.** Giving the frozen model hospital B's own line, set from the scores of 2,000 patients with no labels, cut misses from 484 to 383 of 1,024 but raised false alarms from 15.2% to 28.0%, and utility barely moved (0.265 to 0.271). Run backwards, a 15% budget set at hospital B delivered 45.0% false alarms at hospital A.
- **Local data helps fast, then slows.** On 10,000 hospital B test patients, utility was 0.262 with no local patients in training, 0.326 with 1,000, and 0.385 with 10,000. The cutoff for this curve was set on the same patients it scored, so it shows the shape only and is not a score to quote.
- **Three ideas failed the keep rule and were dropped.** The rule: at least 1.5 points more sepsis patients alerted by onset, and utility not down by more than 0.01. Signature inputs (a summary of each signal's recent shape) scored 0.3994 against 0.3993 at hospital A. A hospital B only model scored 0.394 against the both-hospitals model's 0.406. Training on a score-shaped number in place of the yes or no label scored 0.374 against 0.399 at hospital A.
- **How sure these numbers are.** Resampling patients 1,000 times puts the both-hospitals model at 0.402 at hospital A (95% range 0.381 to 0.422) and 0.406 at hospital B (0.383 to 0.429). Against each hospital's own model the difference is too small to call: the range includes zero at hospital A and touches it at hospital B.

## Where to look

- **[sepsis_early_warning.ipynb](sepsis_early_warning.ipynb)**: the whole project top to bottom, with outputs saved. It retrains the models and reproduces the frozen numbers.
- **[WRITEUP.md](WRITEUP.md)**: the one page write-up ([PDF](WRITEUP.pdf)).
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
- **Challenge paper:** Reyna MA, Josef CS, Jeter R, et al. "Early Prediction of Sepsis from Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019." Critical Care Medicine, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC6964870/
- **Scoring:** the challenge organizers' own scoring script, unchanged, BSD 2-Clause license, in `third_party/`. Copied from https://github.com/physionetchallenges/evaluation-2019
- **Pretrained models:** none.
- **Libraries:** [pandas](https://pandas.pydata.org), [NumPy](https://numpy.org), [scikit-learn](https://scikit-learn.org), [Matplotlib](https://matplotlib.org), [Plotly](https://plotly.com/python/), [Gradio](https://www.gradio.app), [Jupyter](https://jupyter.org).
- **Tutorials:** the hackathon organizers' starter notebook, sent to registered teams. It is the reason this project uses scikit-learn.
- **AI tools:** [Claude Code](https://claude.com/claude-code) wrote most of the code from Tanner's decisions and ran the literature checks. [ChatGPT (Codex)](https://openai.com/codex/) was asked for an outside review. Tanner chose the question, the user, the alert budget and the definitions, adopted the prediction before any model was trained, and decided what to keep.
