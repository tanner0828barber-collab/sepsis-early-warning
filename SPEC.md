# One page spec

Written Sat 11/14/2026, before any code or model. Veritas AI International High School Hackathon, Health track. Solo entry.

**Working name:** Sepsis Early Warning: does it travel?

**Track and brief:** Health. "Build an AI tool that helps one specific person catch a health problem sooner, or manage an ongoing one better."

**User:** an intensive care nurse deciding which patient to look at first.

**Their problem in one sentence:** a nurse covering several patients cannot study every chart every hour, and sepsis can build for hours before anyone names it.

**What they do today instead:** bedside screening rules such as SIRS, plus their own judgment.

**Data:** PhysioNet / Computing in Cardiology Challenge 2019, training sets A and B. Public and de-identified. 40,336 intensive care patients and 1,552,210 hourly rows across two hospitals. License: CC BY 4.0. Source: https://physionet.org/content/challenge-2019/1.0.0/

**Backup data:** none. If the download site is down, the notebook uses a local cached copy of the same files.

**Dumb baselines:** never alert. Also T0, a model that sees only hours in the unit.

**Rival rule:** a modified SIRS comparator. Alert when two or more hold: heart rate over 90; temperature over 38 or under 36 degrees C; breathing rate over 20 or PaCO2 under 32; white cell count over 12 or under 4. It is "modified" because this data has no band count. qSOFA cannot be computed here because mental status is not recorded.

**Main method:** gradient boosting that accepts empty cells directly, trained on hospital A only. Four planned versions with the same settings:

| Name | Inputs | What it tells me |
|---|---|---|
| T0 | Hours in the unit only | How much is just "sepsis tends to show up later in a stay" |
| M1 | Physiology and trends | The body's signals |
| M2 | M1 plus hours in the unit | What time adds |
| M3 | M2 plus measurement-process signals | The full model. The gap from M2 shows dependence on care patterns, not a cause |

**Fresh test:** train and tune on hospital A, freeze the pipeline and the alert threshold, then evaluate once on hospital B. Hospital B is a site-held-out evaluation set, not a blinded test. See DEFINITIONS.md for the disclosure.

**My prediction for the fresh test, before running it:** in DECISIONS.md, committed before any model is trained.

**Alert budget:** at most 15% of hospital A dev patients who never develop sepsis are ever alerted. The threshold is set once to meet this, then frozen.

**Metrics that matter and why:** per patient: caught, missed, false alarms, and hours of warning. Workload: alerts per 1,000 patient-days. Also the challenge's official utility score. Accuracy is not reported as a headline because only 2.17% of hospital A rows are positive, so "never alert" would look about 98% accurate while catching nobody.

**Worse mistake:** a miss, because a missed sepsis patient loses hours of treatment. False alarms still cost attention, which is why the budget caps them at 15%.

**Wow moment in the demo:** replay a real, de-identified hospital B patient hour by hour and watch the alert fire before onset. Then one click to a patient the model misses.

**What I will NOT build:** a diagnostic tool, free entry of patient data, a neural network, the challenge winner's signature features, a third hospital, or anything tuned on hospital B's answers.

**What I can drop without breaking the claim:** T0, M1 and M2, the label-free travel attempts, bootstrap ranges, the hours-of-warning chart, and every Sunday item in the plan.

This project is a prototype and is not medical advice.
