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
- Sat 11/14, 2:21 PM. Found by looking at hospital A: every sepsis patient has between 6 and 10 positive rows, so records stop a few hours after onset. A record ending is not something the model can see.
- Sat 11/14, 2:24 PM. Settings search fixed before running it: six candidates in src/train.py, checked every 50 trees up to 600. Judged on the tune pile by the share of sepsis patients alerted at or before onset, at the 15% budget. Rule: take the best score, and among candidates within 1 percentage point of it keep the smallest model. Reason for the tie rule: the tune pile has 215 sepsis patients, so small differences are noise, and a smaller model is less likely to have memorized hospital A.
- Sat 11/14, 2:24 PM. Modified SIRS uses the latest known value of each input, carried forward however old. A value never measured counts as "does not hold."
- Sat 11/14, 2:24 PM. A sepsis patient alerted only after onset counts as missed. We keep a separate count of those.
- Sat 11/14, 2:33 PM. Settings search finished (8 minutes, 72 tries, results/settings_search.csv). Best tune score was 65.1% of sepsis patients alerted by onset. The written rule picked "base weighted" with 150 trees (64.7%), the smallest model within 1 point of the best. Honest note: the tune pile has 215 sepsis patients, so every candidate is within noise of the others on this number. A different candidate ("slow careful") had the best hourly AUROC and AUPRC. We followed the rule written before the search, not the result we liked after it.
- Sat 11/14, 2:36 PM. Fixed a flaw found on dev: when many patients tie on their top score (the time-only model T0), the threshold let more than 15% through. The cutoff now leaves tied patients out, so the budget is never exceeded. Test added. M3's numbers did not change.
- Sat 11/14, 2:37 PM. Hospital A dev scored (results/a_dev_results.json). M3 at the 15% budget: alerted by onset on 219 of 358 sepsis patients (58 caught inside the 12 hour window, 161 very early), 139 missed, 556 of 3,710 false alarms, utility 0.358. Modified SIRS: 313 of 358 alerted by onset but 2,916 of 3,710 false alarms, utility 0.090.
- Sat 11/14, 2:38 PM. Found on hospital A dev: time in the unit alone (T0) scores utility 0.312, more than body signals alone (M1, 0.177). Only 1 of 3,710 dev patients without sepsis has a stay of 72 hours or more, while a quarter of sepsis patients have their label start after hour 71. So in this dataset a long stay nearly gives the answer away. The model alerts by onset on 84% of sepsis patients whose label starts at hour 24 or later and only 34% of those whose label starts earlier.
- Sat 11/14, 2:40 PM. Tanner chose to improve the model before freezing. Nothing was frozen and hospital B is still unopened.

## Improvement plan (written at 2:42 PM, before any of it was run)

What the research says helped on this dataset: the first place team (Morrill et al.) added shock index, the BUN to creatinine ratio, a partial SOFA score, counts of measurements in a look-back window, and the highest and lowest of each vital sign over a window. Yang et al. 2020 used 168 inputs and found time in the unit was their single most important input, which matches what we saw on dev.

Rules, fixed now:
1. **Every choice is made by 3-fold cross-validation over the fit and tune patients together** (16,268 patients, 1,432 with sepsis), split by patient. That is about seven times more sepsis patients than the tune pile alone, so the yardstick is much less noisy.
2. **Dev is not used to choose anything.** We already looked at dev once, for the first model. It gets scored one more time, for the final model only.
3. **Yardstick:** share of sepsis patients alerted at or before onset at the 15% budget. Same yardstick as the first search.
4. **Keep rule:** a change is kept only if it raises the yardstick by at least 1.5 percentage points and does not lower the official utility score by more than 0.01. Otherwise the simpler version stays.
5. **Time box:** improvement stops at 3:35 PM. Whatever is best by the rules at that point gets frozen.

Steps, in order:
1. Measure the current model under cross-validation (the baseline).
2. New inputs. Body signals: shock index, pulse pressure, BUN to creatinine ratio, partial SOFA (4 of 6 parts: platelets, bilirubin, blood pressure, creatinine), partial qSOFA (2 of 3 parts), SIRS count, spread of each vital over 6 hours, 24 hour mean, lowest and highest of each vital, and the change between each lab's last two measurements. Measurement signals: how many different labs so far, labs per hour so far, hours since any lab, and empty vital cells in the last 6 hours.
3. Settings: the current pick, "slow careful," and "slow careful" with weighting, each checked every 50 trees up to 600. Among tries within 0.5 points of the best, the smallest model wins.
4. A wider training target: teach the model to recognize the 12 hours before onset, not only the 6 the label marks.
5. Alert smoothing: alert on the average of the last 2 or 3 hourly scores, so one noisy hour cannot fire an alert.
6. Retrain the winner on fit and tune together, score dev once, freeze.

Added at 2:52 PM, after steps 1 and 2 came back and before running it:

7. Grade each patient against other patients at the same hour of their stay. The first model alerts by onset on 84% of sepsis patients whose label starts at hour 24 or later and only 34% of the earlier ones, because its score climbs with time in the unit. Replacing the score with "the share of all patients at this same hour who score lower" might spend the alert budget more evenly across the stay. It needs no sepsis labels. Same data, yardstick and keep rule. Honest note: this idea came from looking at dev results for the first model, so it is not a blind idea. It is still judged only by cross-validation.

## Improvement round results (finished 3:04 PM)

All numbers are 3-fold cross-validation on 16,268 fit and tune patients (1,432 with sepsis), at the 15% budget. Full log: results/improvement_log.csv, results/improvement_settings.csv, results/improvement_hour_adjust.csv.

| Step | Sepsis patients alerted by onset | Utility | Kept? |
|---|---|---|---|
| 1. Baseline: first model's settings | 59.4% | 0.343 | starting point |
| 2. 64 added inputs (164 in all) | 60.3% | 0.347 | No. Gain of 0.8 points, under the 1.5 point rule |
| 3. Settings: "slow careful," 200 trees | 61.1% | 0.356 | Yes. Gain of 1.7 points |
| 4. Wider training target | 60.3% | 0.365 | No. Lower than step 3 |
| 5. Smoothing over 2 hours | 61.5% | 0.366 | No. Gain of 0.4 points |
| 5. Smoothing over 3 hours | 60.8% | 0.369 | No |
| 7. Same-hour grading | 46.3% | 0.086 | No. Much worse |

- Sat 11/14, 3:04 PM. Final choice by the rules: the original 100 inputs, "slow careful" settings (learning rate 0.03, 31 leaves, at least 200 rows per leaf, L2 of 10, no class weighting), 200 trees, the challenge's own label, no smoothing.
- Sat 11/14, 3:04 PM. What the round taught us. The model is on a plateau: five reasonable ideas moved the yardstick by less than 2 points each. The added inputs did not help because the trees can already build those combinations from the raw values. Same-hour grading made things much worse, which confirms that a large part of the model's skill is time in the unit: take time away and it drops from 61% to 46%.
- Sat 11/14, 3:07 PM. Final models trained on fit and tune patients together (16,268) and scored on dev. This was the second and last look at dev. M3: alerted by onset on 222 of 358 sepsis patients (60 caught inside the 12 hour window, 162 very early), 136 missed, 556 of 3,710 false alarms, utility 0.368, hourly AUROC 0.834. The first model had 219, 139, 556, 0.358 and 0.824. The gain is real in cross-validation and small on dev, inside the dev range (57% to 67%).
- Sat 11/14, 3:07 PM. On dev, M2 (no measurement signals) scores utility 0.367 and M3 scores 0.368. At hospital A the measurement signals add almost nothing to utility. M1 (body signals only) scores 0.197 and T0 (time only) 0.312.
- Sat 11/14, 3:10 PM. Tanner asked for an outside review by a second AI tool (ChatGPT in Codex) and chose to freeze and run hospital B now, with any further improvement to come after. Consequence, agreed before the run: any change to the model made after the hospital B run is after-the-fact, gets labeled that way, and can never replace the frozen main result. The reviewer is barred from using hospital B for any experiment.
- Sat 11/14, 3:11 PM. FROZEN. results/frozen.json holds the settings, the 100 inputs, and every threshold, all set on hospital A. M3 alert line: set on dev at the 15% budget. Also frozen in advance for the one run: T0, M1, M2, the modified SIRS rule, the 5%, 10% and 20% budgets, and two label-free travel attempts (same share of all patients alerted, and values rescaled by each hospital's own average and spread).
- Sat 11/14, 3:13 PM. THE ONE HOSPITAL B RUN done (commit deb692a). Numbers frozen. See "Result vs prediction" at the bottom.
- Sat 11/14, 3:50 PM. Tanner asked to intensify training on patients aged 40 and over, after seeing that 89% of all patients are 40 or older. Written before running it: this is an after-the-fact idea (hospital B has been scored), so it cannot replace the frozen result. Test on hospital A only, same 3 folds, same yardstick and keep rule as the improvement round. Two versions: (a) rows from patients 40 and over count double in training, (b) train only on patients 40 and over. Report results for all patients and separately for under 40, 40 and over, and 80 and over, because the Health track asks whether the model works equally well across age groups. Claude's expectation, logged in advance: little change overall, since 40 and over is already 89% of the training data, and a risk of doing worse for patients under 40.
- Sat 11/14, 3:54 PM. Result of the age 40 and over test (results/after_the_fact_age_weight.csv, 3-fold cross-validation on 16,268 hospital A patients). Sepsis patients alerted by onset, all ages: 61.1% with the frozen settings, 60.6% when rows from patients 40 and over count double, 61.0% when training only on patients 40 and over. For the 40 and over group itself: 61.1%, 60.1%, 60.6%. Neither version passes the keep rule, and training only on 40 and over raised false alarms for patients under 40 from 14.9% to 17.4%. Dropped. Reason it could not help: patients 40 and over are already 89% of the training data, so the model was already built mostly from them.
- Sat 11/14, 4:04 PM. After-the-fact, hospital A only: measured how utility and catches change with the alert budget, using held-out scores for the frozen settings on 16,268 fit and tune patients (results/after_the_fact_budget_curve_A.csv). Utility is 0.356 at the 15% budget, 0.397 at 25%, and peaks at 0.422 at 40%. Sepsis patients alerted by onset go from 61% to 71% to 84%, and false alerts per true alert from 2.5 to 3.6 to 4.9. The 15% budget, not the model's ranking, is what held utility down. Caveat from the outside review: this table sets each cutoff on the same patients it scores, so treat it as a guide and confirm with nested cross-validation before choosing a new operating point.
- Sat 11/14, 4:02 PM. Outside review by ChatGPT (Codex) received. It found no ordering violation in the frozen run. Its main points: lead with the 109 of 1,142 timely catches, not the 486 alerted by onset; the improvement round set its cutoff on the same held-out patients it scored (its nested rerun reproduced our baseline: utility 0.355, 61.1%); hospital A dev numbers are calibration numbers because the threshold was set there; rename "relative risk score"; finish the limits card. Its two hospital A experiments (a utility-window training target, and dropping measurement inputs) both failed the keep rule.
- Sat 11/14, 4:08 PM. Tanner set the false alarm cap for version 2 at 25%, to raise utility and cut missed patients. Version 1 stays at 15% and stays the frozen result. Everything in version 2 is after-the-fact and will be labeled that way.
- Sat 11/14, 4:20 PM. Scouting on hospital A (after-the-fact, cutoffs set on the scored patients, so a guide for ranking ideas and not a result; files results/scouting_fewer_misses_A.csv and results/scouting_ranking_ideas_A.csv). At the 25% cap the frozen settings give utility 0.397 with 409 of 1,432 sepsis patients missed; 308 of those 409 are day 1 or late-entry cases. Ideas that did not beat it: a SIRS backstop in the first 6 hours (0.375, 426 missed), a separate day 1 share of the alert budget (425 to 516 missed), a specialist model for hours 1 to 24 (0.388, 411), counting reward-window rows three times (0.392, 409), and averaging three models (0.398, 401, inside the noise). One thing that did matter: training on half the patients gives 0.375 and 422 missed, so more training data still helps.
- Sat 11/14, 4:20 PM. Tanner asked whether limiting the tool to patients 40 and over would give a higher utility. On hospital A at the 25% cap: 0.399 for 40 and over only against 0.397 for all ages. No real gain, and it would drop 11% of patients, so the model stays all ages.
- Sat 11/14, 4:22 PM. Tanner shared a review of sepsis biomarkers (Faix, 2013). The markers it covers that exist in this dataset (lactate, bilirubin, creatinine, platelets, fibrinogen, white cells) are already model inputs. The others (procalcitonin, C-reactive protein, IL-6, CD64 and more) are not in the data.

## Why no new outside data

The other open intensive care datasets either need a credentialed application that takes days, or have no ready-made hourly sepsis label. Building that label ourselves is days of careful work and easy to get wrong. A third hospital is listed as future work.

## Result vs prediction

The one hospital B run finished at 3:13 PM on Sat 11/14 (commit deb692a), using the models and thresholds frozen at 3:10 PM (commit c3fbed6). Hospital B: 20,000 patients, 1,142 with sepsis.

| Prediction | What happened | Verdict |
|---|---|---|
| 1. Utility goes down, by a third to a half | 0.368 at A dev to 0.238 at B (95% range 0.214 to 0.262). A drop of 35% | Right |
| 2. False alarms go above the 15% budget (low confidence) | 1,358 of 18,858, which is 7.2%. Half the budget, not more | Wrong |
| 3. M3 loses more than M1 when it travels | M3 utility 0.368 to 0.238 (down 35%). M1 0.197 to 0.091 (down 54%). Alerted by onset: M3 62% to 43%, M1 53% to 28% | Mostly wrong. M3 lost a little more utility in absolute terms (0.130 against 0.106), but M1 lost more of what it had and more catches |

What we did not predict:
- The model got quieter at hospital B, not noisier. Scores ran lower there, so the frozen alert line was crossed less often, for sepsis patients as well. Catches by onset fell from 222 of 358 (62%) to 486 of 1,142 (43%).
- Time alone matched the full model. T0 scored utility 0.242 at B and M3 scored 0.238.
- Alerting on the same share of all patients as at A dev (a label-free adjustment) brought catches by onset back to 55% (629 of 1,142) at a 17.1% false alarm rate, utility 0.270.
- Late-entry cases: M3 caught 14 of 223. The modified SIRS rule caught 150 of 223.
- The modified SIRS rule at B: alerted by onset on 966 of 1,142, false alarms on 12,630 of 18,858 (67%), utility -0.014.

Why we think prediction 2 was wrong: PENDING, Tanner's read after the five failures.
