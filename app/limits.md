# Limits and model card

**This project is a prototype and is not medical advice.**

## What it is

A retrospective risk-prioritization demo. Every hour it gives each intensive care patient a model score for sepsis and raises an alert the first time the score crosses a frozen line. The score is not a probability. It never says a patient has or does not have sepsis.

- **Built:** November 14 to 15, 2026, in a 48 hour hackathon, by one high school student with AI coding help.
- **Intended user:** an intensive care nurse deciding which patient to look at first.
- **Not intended for:** diagnosis, treatment decisions, or any use on real patients.

## Data

- **Source:** PhysioNet / Computing in Cardiology Challenge 2019, training sets A and B. Public, de-identified, license CC BY 4.0.
- **Hospital A** (20,336 patients) was used to build the model. **Hospital B** (20,000 patients) was scored once, after everything was frozen.
- **One row** is one hour of one patient's stay: 7 vital signs, 26 lab values, age, sex, and hours in the unit.

## The label and the six hour shift

The challenge defines a sepsis time from clinical suspicion of infection (the earlier of antibiotics being ordered and blood cultures being drawn) and organ damage (a two point rise in the SOFA score). The label in the data switches on 6 hours before that time. I call the challenge's time "onset," so onset is the first labeled hour plus 6.

Because the label is built partly from when staff ordered antibiotics and cultures, it depends on how quickly a hospital's clinicians act. The model partly learns that.

## What counts as what

- **Caught:** a sepsis patient whose first alert comes in the 12 hours before onset.
- **Very early:** a sepsis patient whose first alert comes more than 12 hours before onset. Reported separately, because an alert a day and a half early is closer to "high risk patient" than "sepsis is starting."
- **Missed:** a sepsis patient with no alert by onset.
- **False alarm:** a patient who never develops sepsis and is alerted at least once.

## Where the alert line came from

It was set once, on 4,068 hospital A patients the model did not train on, so that at most 15% of patients who never developed sepsis were ever alerted. Then it was frozen and committed before hospital B was opened. It was not moved afterward. Because the line was set on those patients, hospital A dev is the calibration set: its numbers describe the operating point and are not an independent test.

## Hospital B is not a blind test

Before any modeling, hospital B was summarized: patient and row counts, sepsis prevalence, positive-row share, stay length, age and sex mix, sepsis rate by age and by sex, and how empty four columns are. Two facts (how often a lactate value is present, and the unit hour of positive rows) were computed on both hospitals pooled, and they helped motivate two input groups. No model, input selection, or threshold ever used hospital B.

So hospital B is a **site-held-out evaluation set, not a blinded test**. A truly unseen hospital is future work.

My hospital A to hospital B test is also not the challenge's hidden hospital C test. My numbers are not comparable to the challenge leaderboard.

## What the measurement signals do and do not mean

The full model sees how long ago each value was measured and how many lab values appeared recently. A lab value appearing is a sign that something was measured. It is not proof a test was ordered at that moment or that a clinician suspected sepsis. These signals describe a hospital's habits, which is exactly why they may not travel.

## Known weak spots

RESULTS_PLACEHOLDER

## Would this be safe to use?

**No.** It would take, at least: a prospective test at the hospital that would use it, with its own alert line; a check that staff can act on an alert and are not worn down by false ones; calibration, so the score means something as a number; and a much wider fairness audit than age and sex.

## Credits

- Data and challenge: Reyna MA, Josef CS, Jeter R, et al. "Early Prediction of Sepsis from Clinical Data: the PhysioNet/Computing in Cardiology Challenge 2019." Critical Care Medicine, 2020.
- Scoring: the challenge organizers' own scoring script, unchanged (BSD 2-Clause license).
- Libraries: pandas, NumPy, scikit-learn, Plotly, Gradio.
- AI tools: Claude Code wrote most of the code and ran literature checks. ChatGPT (Codex) gave an outside review after the freeze. Tanner chose the question, the user, the alert budget and the definitions, and decided what to keep.
