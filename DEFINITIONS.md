# Definitions

Fixed Sat 11/14/2026, before any code. Every number in this project uses these meanings.

## Time

- **Hour:** one row of a patient's record. The data has one row per hour in the intensive care unit.
- **Label start:** the first hour a patient's sepsis label is 1.
- **Onset:** the challenge's defined sepsis time. The challenge switched the label on six hours before it, so onset = label start + 6 hours, when the record goes back that far.
- **Late-entry cases:** patients whose label is already on in their first row. They have no pre-label history in the record. They are reported separately, and their warning time is computed where it can be computed. I do not say an alert is "impossible" for them.

## Alerts

- **Model score:** the model's output for one patient at one hour. It is a ranking score, not a probability. Calibration was not checked. The app shows it divided by the frozen alert cutoff, so 1 is the alert line. (Until Sat 11/14 4:40 PM this was called "relative risk score." It was renamed after the outside review because "risk" sounds like a probability. The meaning did not change.)
- **Threshold:** one score cutoff, chosen once on hospital A dev patients to meet the alert budget, then frozen and applied unchanged to hospital B. That makes hospital A dev the calibration set: its false alarm share is 15% by construction.
- **Alert budget:** at most 15% of hospital A dev patients who never develop sepsis are ever alerted.
- **Alert:** the first hour a patient's score crosses the frozen threshold.
- **Repeat alerts:** later separate crossings for the same patient.
- **Hours of warning:** onset minus the first alert hour. I report the whole distribution, not only "caught before onset."

## Outcomes, counted per patient

- **Caught:** a sepsis patient whose first alert comes at or before onset and no more than 12 hours before onset. The challenge's reward window begins 12 hours before onset.
- **Very early:** a sepsis patient whose first alert comes more than 12 hours before onset. Reported separately from caught.
- **Missed:** a sepsis patient with no alert by onset.
- **False alarm:** a patient who never develops sepsis and is alerted at least once.

Missed cases and false alarms are always reported separately, never folded into one accuracy number.

## Wording rules

- A lab value appearing in a window is a sign that something was measured. It is not proof a test was ordered at that moment or that a clinician suspected sepsis.
- The tool is a retrospective risk-prioritization demo. It never says a patient has or does not have sepsis.
- My hospital A to hospital B test is not the challenge's hidden hospital C test. My numbers are not comparable to the challenge leaderboard.

## Disclosure: hospital B is not a blind test

Before any modeling, hospital B was summarized: patient and row counts, sepsis prevalence, positive-row share, stay length, age and sex mix, sepsis rate by age and by sex, and how empty four columns are. Two label-split facts (how often a lactate value is present, and the unit hour of positive rows) were computed on hospitals A and B pooled, and they helped motivate two feature groups. No model, feature selection, or threshold has ever used hospital B.

So hospital B is a site-held-out evaluation set, not a blinded test. From this commit on, hospital B is not opened, summarized, or plotted until the one frozen run. All development uses hospital A only. A truly unseen site is future work.

This project is a prototype and is not medical advice.
