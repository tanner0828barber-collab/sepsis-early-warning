# Sepsis Early Warning

**An hour by hour sepsis alarm for intensive care nurses, tested at a hospital it had never seen.**

Tanner Barber, solo. Health track. This project is a prototype and is not medical advice.

| 7.2% vs 67% | 486 of 1,142 | 0.40 vs 0.43 | 40,336 |
|---|---|---|---|
| false alarms: my model vs the standard bedside rule, at a new hospital | sepsis patients flagged by the time sepsis began, at that new hospital | my score with local data vs the 2019 challenge winner (1 is perfect) | patient records, two hospitals, one frozen test run, solo in 48 hours |

## The problem
I chose this problem because my great uncle is in the hospital with sepsis right now. About 1.7 million U.S. adults develop sepsis each year, and at least 350,000 die in the hospital or are discharged to hospice ([CDC](https://www.cdc.gov/sepsis/about/)). Alarms built at one hospital often stumble at the next. Checked at an outside hospital, the widely used Epic Sepsis Model missed 67% of sepsis patients and alerted on 18% of all hospitalized patients ([Wong et al., 2021](https://pmc.ncbi.nlm.nih.gov/articles/PMC8218233/)).

## What I built
I built a model that reads an intensive care record hour by hour (7 vital signs, 26 labs, age, sex, hours in the unit) and alerts when its score crosses a fixed line. **The user is an intensive care nurse deciding which patient to look at first.**

- **Replay:** a real, de-identified stay hour by hour, with live vital signs, the score against the alert line, and the moment it fires. The outcome stays hidden until the replay gets there.
- **Unit board:** 12 beds ranked by score each hour, so the patient to look at first is on top.

Try it: `app/app.py` ([steps](https://github.com/tanner0828barber-collab/sepsis-early-warning#run-it)), the [notebook](https://github.com/tanner0828barber-collab/sepsis-early-warning/blob/main/sepsis_early_warning.ipynb), and the [code](https://github.com/tanner0828barber-collab/sepsis-early-warning).

## What I added
My own idea: my user (an intensive care nurse), my data (public records from two hospitals), and my question (does a model trained at hospital A still work at hospital B?).

## How it works
- **Data:** [PhysioNet Challenge 2019](https://physionet.org/content/challenge-2019/1.0.0/), CC BY 4.0. 40,336 patients, one row per hour. I dropped one column that is 100% empty at hospital A. Reading raw rows, I found records stop a few hours after onset.
- **Split:** hospital A by patient: 13,827 fit, 2,441 tune, 4,068 dev. I set the alert line once on A dev so at most 15% of patients without sepsis get an alert.
- **Test:** I wrote down my predictions, froze everything, then scored hospital B (20,000 patients) exactly once.
- **Models:** never alert, a modified SIRS bedside rule, hours in the unit only, then gradient boosting on 100 inputs.
- **Disclosure:** hospital B is site-held-out, not blind. Its counts, age and sex mix, and sepsis rates were summarized before modeling. No model or alert line ever used it.

## Results

| Method | Utility, A dev | Utility, B (95% range) | Timely at B | Missed at B | False alarms at B |
|---|---|---|---|---|---|
| Modified SIRS rule | 0.090 | -0.014 (-0.051 to 0.023) | 306 | 176 | 12,630 (67.0%) |
| Hours in the unit only | 0.312 | 0.242 (0.217 to 0.268) | 79 | 717 | 2,581 (13.7%) |
| **Full model** | **0.368** | **0.238** (0.214 to 0.262) | **109** | **656** | **1,358 (7.2%)** |

Utility is the challenge's official score (1 is perfect, never alerting is 0). "Timely" is a first alert in the 12 hours before onset.

**What it is good at**

- **Quiet.** At a hospital it had never seen: 1,358 false alarms against the bedside rule's 12,630. The rule scored below never alerting.
- **Still finds patients.** 486 of 1,142 flagged by onset at hospital B, with an hourly AUROC of 0.78.
- **Holds up next to a commercial alarm.** Epic's model in its outside check: AUC 0.63, 67% missed, 18% alerted. Mine: 57% missed, 7.2% false alarms. Different patients and definitions, so this is context, not a head to head test.
- **Close to the challenge winner once it has local data.** This data comes from the PhysioNet 2019 sepsis challenge, where the winning team scored about 0.43 at these two hospitals ([official results](https://moody-challenge.physionet.org/2019/results/challenge.tsv)). Trained on both hospitals after the frozen run, mine scored 0.402 and 0.406, with 25.2% false alarms. After-the-fact, on different patients.

**What it is not good at yet.** Only 109 first alerts landed in the final 12 hours, and 656 patients were missed. At hospital B it did not beat the hours-only model. First-day sepsis is the weak spot: 11% flagged by onset (57 of 514) against 68% later (429 of 628). Patients 80 and over did worst (36%, against 40% to 47% for younger groups). Women 42%, men 43%. The data has no race or ethnicity field.

**Where it fails** (picked by rule): a man of 50 with normal vital signs and no lab until the onset hour. A man of 77 with fever and white cells at 22.4, whom the SIRS rule would have caught. A false alarm on a man of 36 when labs appeared with no vital signs.

**The test nobody asked for.** Before training I predicted utility would fall by a third to a half (right: 35%), false alarms would pass 15% (wrong: 7.2%), and the full model would lose more than a body-signals-only model (mostly wrong).

**Limits.** Retrospective, and never tried by a nurse. The label depends partly on when staff ordered antibiotics and cultures.

## What I'd do next
Set each hospital's alert line from its own unlabeled data (a first try: 629 of 1,142 flagged by onset at 17.1% false alarms), then test at a hospital nobody has looked at.

## Sources
Data: PhysioNet Challenge 2019, CC BY 4.0 (Reyna et al., 2020). Scoring: the organizers' script, unchanged. No pretrained models. **AI tools:** [Claude Code](https://claude.com/claude-code) wrote most of the code and drafted the prediction from published evidence. [ChatGPT (Codex)](https://openai.com/codex/) gave an outside review. I chose the question, user, alert budget and definitions, adopted the prediction before training, and decided what to keep. Every credit with its link: [README](https://github.com/tanner0828barber-collab/sepsis-early-warning#data-and-credits). Every decision: [DECISIONS.md](https://github.com/tanner0828barber-collab/sepsis-early-warning/blob/main/DECISIONS.md).
