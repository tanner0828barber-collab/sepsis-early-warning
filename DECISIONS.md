# Decisions log

Every real choice in this project, with the time it was made (Phoenix time) and the reason. Tanner makes the choices. Claude Code writes most of the code.

## Prediction (written before any model is trained)

### Tanner's prediction

Sat 11/14, 2:20 PM. Tanner read the evidence summary and the prediction below and adopted it as his own before any model was trained. His words: "i agree with that prediction make it mine." So the three points below are the prediction of record for this project. The wording and the sources are Claude's. The decision to stand behind it is Tanner's.

### The prediction (drafted by Claude at about 2:17 PM from published evidence, adopted by Tanner at 2:20 PM)

1. **Utility score at hospital B: down.** The top challenge teams scored about 0.43 on hospitals they trained on and below zero on one they had not seen. Moor et al. 2023 saw AUC fall from 0.846 inside a site to 0.761 across sites. Wong et al. 2021 measured 0.63 for a deployed alert against a vendor-reported 0.76 to 0.83. Guess, not a source: B loses a third to a half of the A dev score, and could go below zero.
2. **False alarms at hospital B: more than the 15% budget, with low confidence.** The direction depends on whether ordinary hospital B patients look sicker to a model raised on hospital A, which nobody can know without looking. What is near certain is arithmetic: sepsis is rarer at B (5.7% against 8.8%), so even at the same false alarm rate a smaller share of alerts will be true.
3. **M3 wins at home and loses more when it travels. M1 loses less.** The challenge paper says entries underperformed on the hidden hospital "particularly if they encoded data collection behaviors specific to a given hospital system." Singh et al. 2021 found missingness features helped in retrospective tests and not in a simulated prospective one. M3 may still score higher than M1 at B in absolute terms.

## Log

- Sat 11/14, 2:00 PM. Project started. No project code existed before this.
- Sat 11/14, 2:07 PM. Question chosen: can a sepsis early warning model trained at hospital A travel to hospital B? User: an intensive care nurse deciding which patient to look at first.
- Sat 11/14, 2:07 PM. Tools: scikit-learn gradient boosting, pandas, Gradio, Plotly. Chose scikit-learn over LightGBM or XGBoost because it is the same kind of model, the organizers' starter notebook uses it, and it adds no install risk. Chose gradient boosting over a neural network because the first and second place challenge teams used boosted trees and it is easier to explain.
- Sat 11/14, 2:09 PM. Alert budget set to 15% of hospital A dev patients who never develop sepsis. Reason, in Tanner's words: "i want more catches but prob like a little below the higher budget." A miss is the worse mistake, but alarm load has to stay under the 20% end of the range.
- Sat 11/14, 2:09 PM. Hospital B stays closed until the one frozen run. All development uses hospital A only.
- Sat 11/14, 2:09 PM. No new outside data this weekend. See the reason in the next section.
- Sat 11/14, 2:20 PM. Split: hospital A patients go into three piles by patient, about 68% fit, 12% tune, 20% dev. Chose three piles over the planned 80/20 because choosing settings on dev would make the dev numbers look better than they are. Dev is used only to set the threshold and report hospital A results.
- Sat 11/14, 2:20 PM. Added the admission-gap column (hours between hospital admission and unit admission) to the time group, so it enters at M2. It is almost never empty at hospital A.
- Sat 11/14, 2:20 PM. The medical versus surgical unit columns are not used. They are empty in 48.9% of hospital A rows.
- Sat 11/14, 2:20 PM. The official utility score will use the challenge organizers' open scoring script, downloaded into this repo with credit.
- Sat 11/14, 2:22 PM. Hospital A cut by patient with seed 2026: 13,827 fit, 2,441 tune, 4,068 dev. Each pile is 8.8% sepsis patients. Saved to splits/hospital_A_piles.csv.
- Sat 11/14, 2:22 PM. Dropped the EtCO2 column. It is empty in 100.0% of hospital A rows, so a model trained there can learn nothing from it. Three other columns are over 99% empty at A (direct bilirubin, troponin, fibrinogen) and are kept, because the model accepts empty cells.
- Sat 11/14, 2:22 PM. Trends are computed for the seven vital signs only: change over 6 hours, plus the mean, lowest and highest measured value in the last 6 hours. Labs are too sparse for trends.
- Sat 11/14, 2:25 PM. Found by looking at hospital A: every sepsis patient has between 6 and 10 positive rows, so records stop a few hours after onset. A record ending is not something the model can see.
- Sat 11/14, 2:30 PM. Settings search fixed before running it: six candidates in src/train.py, checked every 50 trees up to 600. Judged on the tune pile by the share of sepsis patients alerted at or before onset, at the 15% budget. Rule: take the best score, and among candidates within 1 percentage point of it keep the smallest model. Reason for the tie rule: the tune pile has 215 sepsis patients, so small differences are noise, and a smaller model is less likely to have memorized hospital A.
- Sat 11/14, 2:30 PM. Modified SIRS uses the latest known value of each input, carried forward however old. A value never measured counts as "does not hold."
- Sat 11/14, 2:30 PM. A sepsis patient alerted only after onset counts as missed. We keep a separate count of those.

## Why no new outside data

The other open intensive care datasets either need a credentialed application that takes days, or have no ready-made hourly sepsis label. Building that label ourselves is days of careful work and easy to get wrong. A third hospital is listed as future work.

## Result vs prediction

PENDING until after the one hospital B run.
