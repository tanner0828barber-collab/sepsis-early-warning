# Decisions log

Every real choice in this project, with the time it was made (Phoenix time) and the reason. I make the choices. Claude Code writes most of the code.

## Prediction (written before any model is trained)

### My prediction

Sat 11/14, 2:20 PM. I read the evidence summary and the prediction below and adopted it as my own before any model was trained. So the three points below are the prediction of record for this project. The wording and the sources are Claude's. The decision to stand behind it is mine.

### The prediction (drafted by Claude at about 2:17 PM from published evidence, adopted by me at 2:20 PM)

1. **Utility score at hospital B: down.** The top challenge teams scored about 0.43 on hospitals they trained on and below zero on one they had not seen. Moor et al. 2023 saw AUC fall from 0.846 inside a site to 0.761 across sites. Wong et al. 2021 measured 0.63 for a deployed alert against a vendor-reported 0.76 to 0.83. Guess, not a source: B loses a third to a half of the A dev score, and could go below zero.
2. **False alarms at hospital B: more than the 15% budget, with low confidence.** The direction depends on whether ordinary hospital B patients look sicker to a model raised on hospital A, which nobody can know without looking. What is near certain is arithmetic: sepsis is rarer at B (5.7% against 8.8%), so even at the same false alarm rate a smaller share of alerts will be true.
3. **M3 wins at home and loses more when it travels. M1 loses less.** The challenge paper says entries underperformed on the hidden hospital "particularly if they encoded data collection behaviors specific to a given hospital system." Singh et al. 2021 found missingness features helped in retrospective tests and not in a simulated prospective one. M3 may still score higher than M1 at B in absolute terms.

## Log

- Sat 11/14, 2:00 PM. Project started. No project code existed before this.
- Sat 11/14, 2:07 PM. Question chosen: can a sepsis early warning model trained at hospital A travel to hospital B? User: an intensive care nurse deciding which patient to look at first.
- Sat 11/14, 2:07 PM. Tools: scikit-learn gradient boosting, pandas, Gradio, Plotly. Chose scikit-learn over LightGBM or XGBoost because it is the same kind of model, the organizers' starter notebook uses it, and it adds no install risk. Chose gradient boosting over a neural network because the first and second place challenge teams used boosted trees and it is easier to explain.
- Sat 11/14, 2:09 PM. Alert budget set to 15% of hospital A dev patients who never develop sepsis. Reason: a miss is the worse mistake, so I want more catches, but alarm load has to stay under the 20% end of the range.
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
- Sat 11/14, 2:24 PM. A sepsis patient alerted only after onset counts as missed. I keep a separate count of those.
- Sat 11/14, 2:33 PM. Settings search finished (8 minutes, 72 tries, results/settings_search.csv). Best tune score was 65.1% of sepsis patients alerted by onset. The written rule picked "base weighted" with 150 trees (64.7%), the smallest model within 1 point of the best. Honest note: the tune pile has 215 sepsis patients, so every candidate is within noise of the others on this number. A different candidate ("slow careful") had the best hourly AUROC and AUPRC. I followed the rule written before the search, not the result I liked after it.
- Sat 11/14, 2:36 PM. Fixed a flaw found on dev: when many patients tie on their top score (the time-only model T0), the threshold let more than 15% through. The cutoff now leaves tied patients out, so the budget is never exceeded. Test added. M3's numbers did not change.
- Sat 11/14, 2:37 PM. Hospital A dev scored (results/a_dev_results.json). M3 at the 15% budget: alerted by onset on 219 of 358 sepsis patients (58 caught inside the 12 hour window, 161 very early), 139 missed, 556 of 3,710 false alarms, utility 0.358. Modified SIRS: 313 of 358 alerted by onset but 2,916 of 3,710 false alarms, utility 0.090.
- Sat 11/14, 2:38 PM. Found on hospital A dev: time in the unit alone (T0) scores utility 0.312, more than body signals alone (M1, 0.177). Only 1 of 3,710 dev patients without sepsis has a stay of 72 hours or more, while a quarter of sepsis patients have their label start after hour 71. So in this dataset a long stay nearly gives the answer away. The model alerts by onset on 84% of sepsis patients whose label starts at hour 24 or later and only 34% of those whose label starts earlier.
- Sat 11/14, 2:40 PM. I chose to improve the model before freezing. Nothing was frozen and hospital B is still unopened.

## Improvement plan (written at 2:42 PM, before any of it was run)

What the research says helped on this dataset: the first place team (Morrill et al.) added shock index, the BUN to creatinine ratio, a partial SOFA score, counts of measurements in a look-back window, and the highest and lowest of each vital sign over a window. Yang et al. 2020 used 168 inputs and found time in the unit was their single most important input, which matches what I saw on dev.

Rules, fixed now:
1. **Every choice is made by 3-fold cross-validation over the fit and tune patients together** (16,268 patients, 1,432 with sepsis), split by patient. That is about seven times more sepsis patients than the tune pile alone, so the yardstick is much less noisy.
2. **Dev is not used to choose anything.** I already looked at dev once, for the first model. It gets scored one more time, for the final model only.
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
- Sat 11/14, 3:04 PM. What the round taught me. The model is on a plateau: five reasonable ideas moved the yardstick by less than 2 points each. The added inputs did not help because the trees can already build those combinations from the raw values. Same-hour grading made things much worse, which confirms that a large part of the model's skill is time in the unit: take time away and it drops from 61% to 46%.
- Sat 11/14, 3:07 PM. Final models trained on fit and tune patients together (16,268) and scored on dev. This was the second and last look at dev. M3: alerted by onset on 222 of 358 sepsis patients (60 caught inside the 12 hour window, 162 very early), 136 missed, 556 of 3,710 false alarms, utility 0.368, hourly AUROC 0.834. The first model had 219, 139, 556, 0.358 and 0.824. The gain is real in cross-validation and small on dev, inside the dev range (57% to 67%).
- Sat 11/14, 3:07 PM. On dev, M2 (no measurement signals) scores utility 0.367 and M3 scores 0.368. At hospital A the measurement signals add almost nothing to utility. M1 (body signals only) scores 0.197 and T0 (time only) 0.312.
- Sat 11/14, 3:10 PM. I asked for an outside review by a second AI tool (ChatGPT in Codex) and chose to freeze and run hospital B now, with any further improvement to come after. Consequence, agreed before the run: any change to the model made after the hospital B run is after-the-fact, gets labeled that way, and can never replace the frozen main result. The reviewer is barred from using hospital B for any experiment.
- Sat 11/14, 3:11 PM. FROZEN. results/frozen.json holds the settings, the 100 inputs, and every threshold, all set on hospital A. M3 alert line: set on dev at the 15% budget. Also frozen in advance for the one run: T0, M1, M2, the modified SIRS rule, the 5%, 10% and 20% budgets, and two label-free travel attempts (same share of all patients alerted, and values rescaled by each hospital's own average and spread).
- Sat 11/14, 3:13 PM. THE ONE HOSPITAL B RUN done (commit deb692a). Numbers frozen. See "Result vs prediction" at the bottom.
- Sat 11/14, 3:50 PM. I asked to intensify training on patients aged 40 and over, after seeing that 89% of all patients are 40 or older. Written before running it: this is an after-the-fact idea (hospital B has been scored), so it cannot replace the frozen result. Test on hospital A only, same 3 folds, same yardstick and keep rule as the improvement round. Two versions: (a) rows from patients 40 and over count double in training, (b) train only on patients 40 and over. Report results for all patients and separately for under 40, 40 and over, and 80 and over, because the Health track asks whether the model works equally well across age groups. The expectation, logged in advance: little change overall, since 40 and over is already 89% of the training data, and a risk of doing worse for patients under 40.
- Sat 11/14, 3:54 PM. Result of the age 40 and over test (results/after_the_fact_age_weight.csv, 3-fold cross-validation on 16,268 hospital A patients). Sepsis patients alerted by onset, all ages: 61.1% with the frozen settings, 60.6% when rows from patients 40 and over count double, 61.0% when training only on patients 40 and over. For the 40 and over group itself: 61.1%, 60.1%, 60.6%. Neither version passes the keep rule, and training only on 40 and over raised false alarms for patients under 40 from 14.9% to 17.4%. Dropped. Reason it could not help: patients 40 and over are already 89% of the training data, so the model was already built mostly from them.
- Sat 11/14, 4:04 PM. After-the-fact, hospital A only: measured how utility and catches change with the alert budget, using held-out scores for the frozen settings on 16,268 fit and tune patients (results/after_the_fact_budget_curve_A.csv). Utility is 0.356 at the 15% budget, 0.397 at 25%, and peaks at 0.422 at 40%. Sepsis patients alerted by onset go from 61% to 71% to 84%, and false alerts per true alert from 2.5 to 3.6 to 4.9. The 15% budget, not the model's ranking, is what held utility down. Caveat from the outside review: this table sets each cutoff on the same patients it scores, so treat it as a guide and confirm with nested cross-validation before choosing a new operating point.
- Sat 11/14, 4:02 PM. Outside review by ChatGPT (Codex) received. It found no ordering violation in the frozen run. Its main points: lead with the 109 of 1,142 timely catches, not the 486 alerted by onset; the improvement round set its cutoff on the same held-out patients it scored (its nested rerun reproduced my baseline: utility 0.355, 61.1%); hospital A dev numbers are calibration numbers because the threshold was set there; rename "relative risk score"; finish the limits card. Its two hospital A experiments (a utility-window training target, and dropping measurement inputs) both failed the keep rule.
- Sat 11/14, 4:08 PM. I set the false alarm cap for version 2 at 25%, to raise utility and cut missed patients. Version 1 stays at 15% and stays the frozen result. Everything in version 2 is after-the-fact and will be labeled that way.
- Sat 11/14, 4:20 PM. Scouting on hospital A (after-the-fact, cutoffs set on the scored patients, so a guide for ranking ideas and not a result; files results/scouting_fewer_misses_A.csv and results/scouting_ranking_ideas_A.csv). At the 25% cap the frozen settings give utility 0.397 with 409 of 1,432 sepsis patients missed; 308 of those 409 are day 1 or late-entry cases. Ideas that did not beat it: a SIRS backstop in the first 6 hours (0.375, 426 missed), a separate day 1 share of the alert budget (425 to 516 missed), a specialist model for hours 1 to 24 (0.388, 411), counting reward-window rows three times (0.392, 409), and averaging three models (0.398, 401, inside the noise). One thing that did matter: training on half the patients gives 0.375 and 422 missed, so more training data still helps.
- Sat 11/14, 4:20 PM. I asked whether limiting the tool to patients 40 and over would give a higher utility. On hospital A at the 25% cap: 0.399 for 40 and over only against 0.397 for all ages. No real gain, and it would drop 11% of patients, so the model stays all ages.
- Sat 11/14, 4:22 PM. I shared a review of sepsis biomarkers (Faix, 2013). The markers it covers that exist in this dataset (lactate, bilirubin, creatinine, platelets, fibrinogen, white cells) are already model inputs. The others (procalcitonin, C-reactive protein, IL-6, CD64 and more) are not in the data.
- Sat 11/14, 4:40 PM. Reporting fixes after the outside review. No model, threshold, or result changed. The write-up (WRITEUP.md) and the app now lead with the timely number, 109 of 1,142 at hospital B, and show "alerted by onset" second. "Relative risk score" is renamed "model score divided by the frozen alert cutoff" and marked as not a probability. Hospital A dev is labeled the calibration set wherever its numbers appear. The app's limits card now holds real counts, results by age and sex, and late-entry results, all read from the result files. A miss gets its own amber card, not the green one. False alarms per timely catch (12.5) and alerts per 1,000 patient-days (115) are shown next to the false alarm share. Notebook: two text cells edited to match (the rename, the calibration note, and ChatGPT (Codex) in the credits). Its code cells and outputs were not rerun because nothing they compute changed.

## Version 2 (everything here is after-the-fact: hospital B has been scored)

Version 1 above stays the result of record. Version 2 is "the best model I could build" and can never replace it. Each step is written here before it runs.

- Sat 11/14, 4:29 PM. PLAN, written before running: one model trained on both hospitals (script scripts/v2_both_hospitals.py). Why: training on half of hospital A's patients scored lower than training on all of them, so more data should help, and this is how the top challenge teams trained. What it can and cannot claim: it has seen both hospitals, so it says nothing about travel. It answers "what do you get when a hospital has its own labeled data?"
  - Patients: hospital A fit and tune (16,268) plus all of hospital B (20,000). Hospital A dev stays out, so the hospital A side is the same patients as every earlier cross-validation number.
  - Same 100 inputs, same "slow careful" settings, 200 trees. No hospital name is given to the model.
  - Nested cross-validation, the method the outside review asked for: 3 outer folds by patient, balanced by hospital and by sepsis. Inside each outer training set, 2 inner folds produce held-out scores, and the alert cutoff is set on those. The outer fold is then scored once with that cutoff. No patient is scored with a cutoff that saw them.
  - Cutoff: the 25% cap I chose, applied at each hospital separately (each hospital's cutoff is set on that hospital's inner-fold patients). Reason: a cap is a promise to the nurses at one hospital, and one shared cutoff could break it at one site. The single shared cutoff and the 15% budget are also reported, for comparison only.
  - Baseline, run in the same script on the same folds: a hospital A only model, nested, scored on the same hospital A patients.
  - Keep rule, hospital A patients, 25% cap: the both-hospitals model is "better at A" only if it alerts by onset on at least 1.5 points more sepsis patients than the A only model and its utility is not lower by more than 0.01. The hospital B number is reported whatever it is.
  - The expectation, logged in advance and labeled a guess: at hospital A, utility 0.39 to 0.42 at the 25% cap, a gain of under 0.02 over the A only model. At hospital B, utility 0.30 to 0.38, lower than A because 45% of B's sepsis patients have their label start in the first day. If any utility comes back at 0.50 or more, treat it as a leak until proven otherwise.
- Sat 11/14, 4:44 PM. RESULT of the both-hospitals model (results/after_the_fact_v2_both_hospitals.csv, 14 minutes of training, nested cross-validation as planned). After-the-fact, and it cannot claim to travel.

  | 25% cap, each hospital's own cutoff | Utility | Timely (first alert in the 12 hours before onset) | Alerted by onset | Missed | False alarms |
  |---|---|---|---|---|---|
  | A only model, hospital A patients | 0.399 | 304 of 1,432 (21.2%) | 72.7% | 391 | 3,816 of 14,836 (25.7%) |
  | Both-hospitals model, hospital A patients | 0.402 | 295 of 1,432 (20.6%) | 72.1% | 399 | 3,766 of 14,836 (25.4%) |
  | Both-hospitals model, hospital B patients | 0.406 | 364 of 1,142 (31.9%) | 80.7% | 220 | 4,755 of 18,858 (25.2%) |
  | Both-hospitals model, both together | 0.403 | 659 of 2,574 (25.6%) | 76.0% | 619 | 8,521 of 33,694 (25.3%) |

  - Keep rule at hospital A: NOT met. Adding hospital B's patients changed alerted by onset at A by minus 0.6 points (72.1% against 72.7%) and utility by plus 0.003. So for hospital A, the extra data is no better than hospital A's own data. That is a result, not a failure to hide.
  - Hospital B is where it matters: with its own labeled patients in training and its own cutoff, hospital B reaches utility 0.406 and 220 missed of 1,142. Version 1, which never saw hospital B, scored 0.238 with 656 missed there. The two are not the same test: version 1 used a 15% budget and delivered 7.2% false alarms, while this delivers 25.2%. At the 15% cap the both-hospitals model scores 0.386 at B with 319 missed and 15.5% false alarms.
  - The nested check works as intended: the cap was 25% and the held-out false alarm shares came out at 25.2% to 25.7%, slightly over, because each cutoff was set on patients other than the ones scored.
  - Each hospital needs its own alert line even inside one model: hospital B's cutoff came out near 0.025 and hospital A's near 0.040 in every fold. With one shared cutoff (shown for comparison only), hospital A gets 32.9% false alarms and hospital B 19.5%, which breaks the cap at A.
  - The 15% cap A only number, 0.355 and 61.1%, matches the outside review's nested rerun (0.355, 61.1%), so the method reproduces.
  - The logged guess against the result: hospital A was guessed at 0.39 to 0.42 with a gain under 0.02 (right: 0.402, gain 0.003). Hospital B was guessed at 0.30 to 0.38 (wrong: 0.406). Hospital B is not a harder hospital once the model has its data. No utility came near 0.50, so no leak flag.
  - Costs, stated plainly: at the 25% cap that is 5.2 false alarms per alerted sepsis patient at hospital B and 346 alerts per 1,000 patient-days, about three times version 1's 115.
- Sat 11/14, 4:45 PM. PLAN, written before running: the frozen version 1 model with a per-hospital alert line at the 25% cap (script scripts/v2_alert_line_per_hospital.py). This is a second look at hospital B and is labeled that way. The model is not retrained and no hospital B sepsis label is used to set anything.
  - Step 1, hospital A only: set the frozen model's cutoff on hospital A dev at the 25% cap, by the same rule version 1 used at 15%. Note the share of ALL hospital A dev patients alerted at that cutoff.
  - Step 2, hospital B: take the first 2,000 hospital B patients by ID as a burn-in group. Using their scores only (no labels), find the cutoff that alerts on that same share of them. Apply it to the other 18,000 patients and score those 18,000 once.
  - Comparison on the same 18,000 patients: hospital A's 25% cutoff applied unchanged, which is the version 1 recipe at the new cap.
  - The guess, logged in advance: the unchanged cutoff gives about 14% false alarms at B and utility near 0.26. The burn-in line gives 25% to 30% false alarms, 62% to 70% alerted by onset, and utility 0.28 to 0.32. It should land below the both-hospitals model's 0.406, because that model had hospital B's labels and this one does not.
- Sat 11/14, 4:45 PM. PLAN, written before running: run it backwards, once (script scripts/v2_reverse.py). Train on hospital B, test on hospital A. After-the-fact. Purpose: check whether the version 1 finding (the model gets quieter when it travels) is about travel in general or about the direction A to B.
  - Mirror of version 1: hospital B is cut by patient, 80% to train and 20% to set the cutoff (seed 2026, same share of sepsis in each). Same 100 inputs, same settings, 200 trees. Cutoffs set on the 20% at the 15% budget (the mirror of version 1) and at the 25% cap. Then all 20,336 hospital A patients are scored once. Nothing is tuned on hospital A.
  - The guess, logged in advance: the mirror image of version 1. A model raised at hospital B, where labs are rarer and blood pressures read higher, should see hospital A patients as sicker, so false alarms at A should land above the budget (25% to 40% at the 15% budget) and utility at A should be 0.25 to 0.35.
- Sat 11/14, 4:47 PM. RESULT of the per-hospital alert line at the 25% cap (results/after_the_fact_v2_alert_line.csv). Frozen version 1 model, not retrained. A second look at hospital B, after-the-fact.

  | Frozen model, 25% cap | Utility | Timely | Alerted by onset | Missed | False alarms |
  |---|---|---|---|---|---|
  | Hospital A dev (calibration set, cutoff set here) | 0.420 | 78 of 358 | 74.3% | 92 | 927 of 3,710 (25.0%) |
  | Hospital B, 18,000 patients, hospital A's cutoff unchanged | 0.265 | 142 of 1,024 | 52.7% | 484 | 2,585 of 16,976 (15.2%) |
  | Hospital B, same 18,000, its own line from a 2,000 patient burn-in, no labels | 0.271 | 171 of 1,024 | 62.6% | 383 | 4,747 of 16,976 (28.0%) |

  - Hospital A's 25% cutoff alerts on 29.6% of all hospital A dev patients. The burn-in cutoff that alerts on the same share at hospital B is lower (0.028 against 0.039), which fits the version 1 finding that hospital B scores run lower.
  - What the label-free line buys: 101 fewer missed sepsis patients out of 1,024. What it costs: false alarms go from 15.2% to 28.0%, over the 25% cap. Utility barely moves, 0.265 to 0.271. So moving the line trades alarms for catches and does not make the model better at hospital B.
  - Put next to the both-hospitals result: with the line moved but no hospital B labels, utility at B is 0.271. With hospital B's labeled patients in training it is 0.406. The gap is in how the model ranks hospital B patients, not in where the line sits. A new hospital needs its own labeled data.
  - Guess against result: unchanged cutoff guessed at about 14% false alarms and utility near 0.26 (right: 15.2%, 0.265). Burn-in guessed at 25% to 30% false alarms (right: 28.0%), 62% to 70% by onset (right: 62.6%), utility 0.28 to 0.32 (wrong, a little low: 0.271).
- Sat 11/14, 4:47 PM. RESULT of the backwards run (results/after_the_fact_v2_reverse.csv). Trained on 16,000 hospital B patients, cutoff set on 4,000 others, scored once on all 20,336 hospital A patients. After-the-fact.

  | Trained at hospital B | Utility | Timely | Alerted by onset | Missed | False alarms |
  |---|---|---|---|---|---|
  | 15% budget, hospital B calibration set | 0.380 | 58 of 228 | 70.2% | 68 | 563 of 3,772 (14.9%) |
  | 15% budget, hospital A | 0.330 | 338 of 1,790 | 76.4% | 422 | 8,348 of 18,546 (45.0%) |
  | 25% cap, hospital B calibration set | 0.399 | 65 of 228 | 80.7% | 44 | 942 of 3,772 (25.0%) |
  | 25% cap, hospital A | 0.329 | 382 of 1,790 | 86.1% | 248 | 11,914 of 18,546 (64.2%) |

  - It is the mirror image of version 1. Going A to B the model got quieter: a 15% budget delivered 7.2% false alarms. Going B to A it got much noisier: a 15% budget delivered 45.0%. Same cause in both directions: hospital A patients score higher than hospital B patients whichever hospital the model was raised at.
  - So an alert budget set at one hospital does not hold at another, in either direction. Too quiet one way, three times over budget the other way.
  - Utility fell less in this direction (0.380 to 0.330, down 13%, against 35% for A to B), because the utility score punishes misses much more than false alarms.
  - Guess against result: false alarms at A guessed at 25% to 40% (wrong, it was worse: 45.0%). Utility at A guessed at 0.25 to 0.35 (right: 0.330).
- Sat 11/14, 5:05 PM. PLAN, written before running: compact signature inputs (script scripts/v2_signatures.py). After-the-fact, hospital A only. Source of the idea: an outside literature review I ran (ChatGPT deep research). It reports that the challenge winner (Morrill et al.) went from 0.418 to 0.430 in their own cross-validation by adding path signatures of three signals. The paper itself was not opened, so those two numbers are the review's reading.
  - What a signature is here: a short summary of the shape of a signal over the last 7 hours. For partial SOFA, mean arterial pressure, and the BUN to creatinine ratio: how far it moved, the area under its move (did it change early or late in the window), and how jumpy it was (sum of squared hourly changes). Plus, for each pair of the three signals, a term for which one moved first. That is 12 new inputs. The two signals that are not yet model inputs (partial SOFA and the ratio) are added as plain values too, so the candidate has 114 inputs against 100.
  - Honest limit: this is a compact, hand-computed, order 2 version. No signature library is installed. It is not the winner's full method (order 3, lead-lag on every signal, a different model and training target).
  - Every new input uses only the current hour and the six before it. Nothing looks ahead.
  - Test: the same nested cross-validation, the same folds, and the same 25% cap as the hospital A only baseline logged above (utility 0.399, 72.7% alerted by onset, 391 of 1,432 missed). Same settings, 200 trees.
  - Keep rule: at least 1.5 points more sepsis patients alerted by onset (74.2% or more, so 369 or fewer missed) and utility not below 0.389.
  - The guess, logged in advance: it fails the keep rule. Utility within 0.005 of 0.399 and alerted by onset within 1 point of 72.7%. Reason: the same three signals as plain values were among the 64 inputs that added only 0.8 points. About a one in five chance it passes.
- Sat 11/14, 5:10 PM. RESULT of the compact signature inputs (results/after_the_fact_v2_signatures.csv). Nested, hospital A only, 25% cap, same folds as the baseline. The baseline was read back from its saved scores and cutoffs and reproduced exactly (0.399, 72.7%, 391 missed), so the comparison is like for like.

  | Hospital A, 25% cap | Utility | Timely | Alerted by onset | Missed | False alarms |
  |---|---|---|---|---|---|
  | Baseline, 100 inputs | 0.3993 | 304 of 1,432 | 72.7% | 391 | 3,816 of 14,836 (25.7%) |
  | With signatures, 114 inputs | 0.3994 | 300 of 1,432 | 72.4% | 395 | 3,783 of 14,836 (25.5%) |

  - Keep rule: NOT met. Alerted by onset moved by minus 0.3 points and utility by plus 0.0001. Dropped. The 100 input model stays.
  - Guess against result: right. Guessed utility within 0.005 and alerted by onset within 1 point.
  - What it means: like every earlier idea aimed at the model's ranking, it moved the yardstick by under 2 points. The model already has each signal's latest value, 6 hour change, mean, lowest and highest, and trees can build shape from those. Honest limit on the claim: this tested a compact order 2 version, not the challenge winner's full method, so it does not show that full signatures would fail.

- Sat 11/14, 5:07 PM. PLAN, written before running: how much local data does a new hospital need? (script scripts/v2_local_data_curve.py). After-the-fact. It follows from the both-hospitals result: with no hospital B data the model sits near 0.27 there, with all of it 0.406. This measures the steps in between.
  - Hospital B is cut once by patient (seed 2026, same share of sepsis in each half): 10,000 test patients that no arm ever trains on, and 10,000 local patients to draw from.
  - Arms: hospital A's 16,268 fit and tune patients plus the first 0, 500, 1,000, 2,000, 5,000 or 10,000 local patients in one fixed shuffled order, so each arm contains the one before it. One more arm trains on the 10,000 local patients alone, with no hospital A data, to see whether hospital A's data helps hospital B at all. Same 100 inputs, same settings, 200 trees, no hospital name given to the model.
  - Scoring: every arm is scored on the same 10,000 test patients. The cutoff for each arm is set on those test patients at exactly the 25% cap. That is the same shortcut as the earlier scouting tables: the cutoff sees the patients it scores. It is used here because it is identical for every arm, so the arms differ only in how well the model ranks patients. So this curve is a guide to the shape, not a result to quote as a score. Hourly AUROC, which needs no cutoff, is reported next to it.
  - The guess, logged in advance: utility climbs from about 0.30 with no local patients to about 0.39 with 10,000, and half of that gain arrives by 2,000 local patients. The local-only arm lands a little under the arm with hospital A plus 10,000.
- Sat 11/14, 5:16 PM. RESULT of the local data curve (results/after_the_fact_v2_local_data_curve.csv). Scored on the same 10,000 hospital B test patients (571 with sepsis). Each cutoff was set on those test patients at the 25% cap, the same way for every arm, so read the shape and not the level.

  | Trained on | Utility | Hourly AUROC | Timely | Alerted by onset | Missed (of 571) |
  |---|---|---|---|---|---|
  | Hospital A only, no local patients | 0.262 | 0.776 | 84 | 58.1% | 239 |
  | Hospital A + 500 local | 0.297 | 0.800 | 124 | 66.2% | 193 |
  | Hospital A + 1,000 local | 0.326 | 0.816 | 148 | 71.5% | 163 |
  | Hospital A + 2,000 local | 0.341 | 0.823 | 150 | 71.8% | 161 |
  | Hospital A + 5,000 local | 0.368 | 0.840 | 167 | 76.4% | 135 |
  | Hospital A + 10,000 local | 0.385 | 0.849 | 171 | 76.9% | 132 |
  | 10,000 local only, no hospital A | 0.399 | 0.857 | 166 | 77.4% | 129 |

  - Local data helps fast, then slows. The first 1,000 local patients (about 57 of them with sepsis) recover 52% of the utility gap between no local data and 10,000. The first 500 cut misses from 239 to 193.
  - The surprise: a model trained on the 10,000 local patients alone beat the model that also had hospital A's 16,268 (0.399 against 0.385, AUROC 0.857 against 0.849, 129 missed against 132). Once a hospital has enough of its own patients, the other hospital's data does not help and may hurt a little. One test set and one split, so the size of that gap is not settled. It gets a proper nested test next.
  - Guess against result: no local data guessed at about 0.30 (wrong, lower: 0.262). 10,000 local guessed at about 0.39 (right: 0.385). Half the gain by 2,000 (right, it came sooner: 64% by 2,000). Local-only guessed a little under the mixed model (wrong: it was above).
- Sat 11/14, 5:16 PM. PLAN, written before running: a hospital B only model, nested (script scripts/v2_local_only.py). After-the-fact. The curve above suggests each hospital may be better served by a model trained only on its own patients. Hospital A's own-data model is already measured (0.399). This measures hospital B's.
  - All 20,000 hospital B patients, the same outer folds as the both-hospitals run, 2 inner folds to set the cutoff at the 25% cap. Same 100 inputs, settings and 200 trees.
  - Baseline on the same patients and folds: the both-hospitals model at hospital B (utility 0.406, 80.7% alerted by onset, 220 of 1,142 missed), read back from its saved scores.
  - Keep rule: at least 1.5 points more alerted by onset (82.2% or more, 203 or fewer missed) and utility not below 0.396. If it is met, the best model for hospital B is its own. If not, the both-hospitals model stays the best version 2 model.
  - The guess, logged in advance: utility 0.40 to 0.42 at hospital B, a gain too small to meet the keep rule. About a one in four chance it passes.
- Sat 11/14, 5:22 PM. RESULT of the hospital B only model (results/after_the_fact_v2_local_only.csv). Nested, 25% cap, same outer folds. The both-hospitals baseline was read back from its saved scores and reproduced exactly (0.406, 80.7%, 220 missed).

  | Hospital B, all 20,000 patients, 25% cap | Utility | Timely | Alerted by onset | Missed | False alarms |
  |---|---|---|---|---|---|
  | Both-hospitals model | 0.406 | 364 of 1,142 | 80.7% | 220 | 4,755 of 18,858 (25.2%) |
  | Hospital B only model | 0.394 | 359 of 1,142 | 80.6% | 221 | 5,041 of 18,858 (26.7%) |

  - Keep rule: NOT met. Alerted by onset moved by minus 0.1 points and utility fell by 0.012. The both-hospitals model stays the best version 2 model: 0.402 at hospital A and 0.406 at hospital B.
  - The surprise in the curve did not survive the proper test. In the curve, the local-only arm beat the mixed arm on one test set with the cutoff set on the scored patients. Nested, on all of hospital B, the mixed model is level on catches and a little ahead on utility. This is why the cutoff is set on other patients before a result is believed.
  - Read together with the both-hospitals result: hospital B's data did not help hospital A, and hospital A's data helps hospital B a little or not at all. What a hospital mostly needs is its own labeled patients.
  - Guess against result: utility guessed at 0.40 to 0.42 (wrong, lower: 0.394). Guessed it would not meet the keep rule (right).
- Sat 11/14, 5:41 PM. PLAN, written before running: 95% ranges for the version 2 numbers (script scripts/v2_ranges.py). The outside reviewer asked for uncertainty on the version 2 comparisons. No new model and no new choice: this reads the held-out scores and cutoffs already saved and resamples patients 1,000 times (seed 2026), the same method as the version 1 ranges.
  - Ranges for the both-hospitals model at the 25% cap: utility, timely share, alerted by onset, and false alarm share, at hospital A, at hospital B, and together.
  - Paired differences, resampling the same patients for both models: both-hospitals against hospital A only on hospital A patients; hospital B only against both-hospitals on hospital B patients; hospital B's burn-in alert line against hospital A's unchanged cutoff on the 18,000.
  - To pair the hospital B only model, its script is rerun unchanged except that it now also saves which hours it alerted on. It is seeded, so it must reproduce 0.394 and 221 missed exactly. If it does not, that is reported.
  - How a range will be read: if a difference's 95% range includes zero, the two models are not told apart by this data, and the write-up may not call one better.
- Sat 11/14, 5:48 PM. RESULT: 95% ranges for the version 2 numbers (results/after_the_fact_v2_ranges.csv). Patients resampled 1,000 times, seed 2026. The hospital B only script was rerun to save its alert hours and reproduced its first run exactly.

  | 25% cap | Utility | Timely | Alerted by onset | False alarms |
  |---|---|---|---|---|
  | Both-hospitals model, hospital A | 0.402 (0.381 to 0.422) | 20.6% (18.7 to 22.7) | 72.1% (69.8 to 74.4) | 25.4% (24.7 to 26.1) |
  | Both-hospitals model, hospital B | 0.406 (0.383 to 0.429) | 31.9% (29.1 to 34.6) | 80.7% (78.4 to 82.9) | 25.2% (24.6 to 25.8) |
  | Both-hospitals model, both together | 0.403 (0.388 to 0.418) | 25.6% (23.9 to 27.4) | 76.0% (74.3 to 77.6) | 25.3% (24.8 to 25.8) |
  | Hospital A only model, hospital A | 0.399 (0.378 to 0.420) | 21.2% (19.2 to 23.4) | 72.7% (70.5 to 74.9) | 25.7% (25.0 to 26.5) |
  | Hospital B only model, hospital B | 0.394 (0.371 to 0.418) | 31.4% (28.8 to 34.0) | 80.6% (78.4 to 82.8) | 26.7% (26.1 to 27.3) |
  | Frozen model at B (18,000), hospital A's cutoff | 0.265 (0.240 to 0.289) | 13.9% (11.7 to 15.9) | 52.7% (49.7 to 55.7) | 15.2% (14.7 to 15.8) |
  | Frozen model at B (18,000), its own burn-in line | 0.271 (0.244 to 0.296) | 16.7% (14.3 to 19.0) | 62.6% (59.6 to 65.4) | 28.0% (27.3 to 28.7) |

  Paired differences (the same patients redrawn for both models):
  - Both-hospitals minus hospital A only, at hospital A: utility +0.002 (-0.007 to +0.011), alerted by onset -0.6 points (-2.2 to +0.9). Both ranges include zero. At hospital A the two models cannot be told apart.
  - Hospital B only minus both-hospitals, at hospital B: utility -0.012 (-0.023 to -0.000), alerted by onset -0.1 points (-1.9 to +1.7), false alarms +1.5 points (+1.0 to +2.1). On catches they cannot be told apart. On utility the both-hospitals model is ahead by a hair, with a range that reaches zero at its edge, so the honest wording is "about the same, slightly favoring both hospitals."
  - Hospital B's own burn-in line minus hospital A's cutoff, frozen model: utility +0.006 (-0.003 to +0.015) (includes zero), alerted by onset +9.9 points (+8.0 to +11.6), false alarms +12.7 points (+12.3 to +13.3). So moving the line clearly buys catches and clearly costs false alarms, and does not clearly change utility.
  - What the ranges allow me to say: the both-hospitals model scores about 0.40 at each hospital, give or take 0.02. What they do not allow: calling the both-hospitals model better than a hospital's own model.
- Sat 11/14, 5:43 PM. PLAN, written before running: train on a utility-shaped target (script scripts/v2_utility_target.py). After-the-fact, hospital A only. This is the last idea aimed at the model's ranking that has outside evidence behind it: the literature review says the challenge winner trained on it (the paper itself was not opened).
  - What changes: today the model learns a yes or no label. Here it learns a number for every hour: how much the official score gains if the model alerts at that hour compared with staying silent. That number is minus 0.05 for any hour of a patient who never develops sepsis and for hours more than 12 hours before onset, climbs from 0 to 1 across the 12 to 6 hours before onset, and climbs on to 2 by 3 hours after onset. It is worked out from the organizers' own scoring rules, and the script checks it against their scoring function before training.
  - How it differs from the two target ideas that already failed: those kept a yes or no label and only moved or re-counted which hours were yes. This one is a graded number learned by regression, so an hour near onset is worth more than an hour 11 hours before it.
  - Same 100 inputs, same settings, 200 trees, a regressor in place of the classifier. Same nested cross-validation, folds and 25% cap as the hospital A only baseline (0.399, 72.7% alerted by onset, 391 missed).
  - Keep rule: 74.2% or more alerted by onset and utility not below 0.389.
  - The guess, logged in advance: utility moves by 0.00 to plus 0.01, alerted by onset stays within 1.5 points, and the keep rule is not met. About a one in five chance it passes.
- Sat 11/14, 5:51 PM. RESULT of the utility-shaped target (results/after_the_fact_v2_utility_target.csv). Nested, hospital A only, 25% cap, same folds. The target was checked against the organizers' scoring function at every hour of a test patient before training, and matched.

  | Hospital A, 25% cap | Utility | Timely | Alerted by onset | Missed | False alarms |
  |---|---|---|---|---|---|
  | Baseline: yes or no label | 0.399 | 304 of 1,432 | 72.7% | 391 | 3,816 of 14,836 (25.7%) |
  | Utility-shaped target | 0.374 | 250 of 1,432 | 64.7% | 505 | 3,101 of 14,836 (20.9%) |

  - Keep rule: NOT met. It is worse on both counts: alerted by onset down 8.0 points, utility down 0.025. Dropped.
  - One thing that makes the table look worse than it is: this model's nested cutoff landed at 20.9% false alarms, under the 25% cap, so it alerted on fewer patients than the baseline did. As a scouting check, with both cutoffs set on the scored patients at exactly 25%: utility 0.383 against 0.397, alerted by onset 69.0% against 71.4%, missed 444 against 409. So it is still worse at the same alarm rate, by less.
  - Guess against result: wrong. Guessed utility flat to plus 0.01 and catches within 1.5 points. It lost on both. Guessed the keep rule would not be met (right).
  - What it means: for this model and these inputs, the plain yes or no label ranks patients better than the score-shaped number. That closes the list of ranking ideas with outside evidence behind them. Honest limit: the winner paired this target with other choices (a different library, signatures, a cutoff picked to maximize utility with no cap), so this does not show their whole method would fail here.

## A second count of timely alerts (after-the-fact reporting, no model change)

- Sat 11/14, 6:31 PM. PLAN, written before the script ran: count the sepsis patients whose alert was ON during the 12 hours before onset (script scripts/after_the_fact_alert_on_in_window.py). Why: "timely" in this project counts only a patient's FIRST alert. A patient whose first alert came 30 hours early and whose alert was still on at onset is counted as "very early," not timely. The official utility score rewards every alerting hour inside that window, whenever the first alert was, so the two views can disagree and a reader should see both.
  - Definition, fixed here: a sepsis patient counts as "alert on in the window" if the score is at or above the frozen alert line in at least one recorded hour from 12 hours before onset through the onset hour. Same window edges as "timely."
  - Nothing is retrained and nothing is chosen. The script reads the frozen models and frozen thresholds and counts the same alerts a different way. The frozen files are not written. Version 1's numbers do not change, and 109 of 1,142 stays the timely number.
  - Reported for every planned model and the modified SIRS rule, at hospital A dev and hospital B, so the full model's count can be read next to the time-only model's. The script first checks that it reproduces the frozen timely, very early and missed counts, and stops if it does not.
  - Honest note: this is not a blind plan. The measure was defined after hospital B was scored, and the full model's hospital B count (452 of 1,142) was already seen at about 6:10 PM, in a separate read-only check of the project. So it is after-the-fact and is labeled that way wherever it appears. It answers the open "timely coverage" question.
  - The guess for the numbers not yet seen, logged in advance: the time-only model's count equals its alerted by onset count at hospital B (425), because its alert almost never switches off once it is on. The modified SIRS rule lands above 800 of 1,142.
- Sat 11/14, 6:32 PM. RESULT of the alert-on-in-the-window count (results/after_the_fact_alert_on_in_window.csv). After-the-fact reporting, no model change. Before counting, the script reproduced the frozen timely, very early and missed counts for every method at both sites.

  | Hospital B, 1,142 sepsis patients | Timely (first alert in the 12 hours before onset) | Alert on in those 12 hours (95% range) | Very early, and off through the whole window |
  |---|---|---|---|
  | Modified SIRS rule | 306 | 885 (77.5%, 75.0 to 80.1) | 81 |
  | T0: hours in the unit only | 79 | 425 (37.2%, 34.5 to 39.9) | 0 |
  | M1: body signals and trends | 103 | 248 (21.7%, 19.4 to 24.0) | 69 |
  | M2: M1 plus time | 111 | 464 (40.6%, 37.7 to 43.3) | 38 |
  | M3: the full model | 109 | 452 (39.6%, 36.6 to 42.3) | 34 |

  - Hospital A dev (the calibration set), same count: the full model 206 of 358 (57.5%, 52.8 to 62.8), the time-only model 145 of 358 (40.5%, 35.5 to 45.5), the modified SIRS rule 280 of 358 (78.2%).
  - What it means: both counts are true and they answer different questions. 109 of 1,142 got a NEW alert in the 12 hours before onset. 452 of 1,142 had an alert showing at some hour in those 12 hours, because 343 of the 377 "very early" patients were still alerting or alerting again. For those 343 the alert had first fired more than 12 hours earlier, so it was not news at the bedside. The timely number stays 109. The new count is reported next to it and never in place of it.
  - It does not rescue the model. By this count the time-only model has 425 at hospital B against the full model's 452, and the two ranges overlap (no paired test was run). At hospital A dev the gap is wide, 206 against 145. That is the same story the utility score tells: ahead of the clock at home, about level with it at hospital B.
  - Guess against result: the time-only model guessed at 425 (right). The modified SIRS rule guessed above 800 (right: 885).
  - Where it is reported: the write-up and the README. Not yet in the app or the notebook.

## The unit board (a second screen in the app, no model change)

- Sat 11/14, 6:48 PM. PLAN, written before any patient was picked: a "Unit board" tab in the app (code in app/unit_board.py, patients picked by scripts/build_unit_board.py). Why: my user is a nurse deciding which patient to look at first, and the Replay shows one patient at a time. The board shows 12 hospital B patients at the same hour of their stay, ranked by the frozen model's score, highest first.
  - No model, threshold or result changes. It uses the same frozen model and the same frozen alert line as the Replay. It is a display, not a result, and no number from it is reported as a result.
  - Patients are picked by rule and not by eye: one seeded random draw (seed 2026) from all 20,000 hospital B patients, 9 who never develop sepsis and 3 who do. The draw never reads the model's scores or outcomes, so nobody chose patients the model does well or badly on. Whatever the draw gives stays, including if the model misses all three.
  - Disclosed on the screen: 3 of 12 is far more sepsis than the real share at hospital B (1,142 of 20,000, about 1 in 18). Sepsis patients were added on purpose so the board has something to find. The 12 patients were not in the unit together: the board lines their records up by hour in the unit. Bed numbers are made up and come from the same seeded shuffle, so a bed number says nothing about the patient.
  - What happened to each patient stays hidden until the board's hour reaches it, the same rule as the Replay.
  - Words on the screen: "Over the line" is a score at or above the frozen alert line, "Near the line" is half the line or more, "Below the line" is anything lower. Half is a display choice, not a clinical standard and not a second alert line.
  - Choices made to keep moving, which I can still change: 12 beds, the 9 and 3 mix, the "near" level at half the line, the board opening at hour 12, and the board being the first tab.
- Sat 11/14, 6:55 PM. RESULT: the unit board is built and is the first tab of the app. The draw ran once and was kept as drawn. No model, threshold or reported number changed.
  - Who was drawn: 12 hospital B patients. The 3 who develop sepsis sit in beds 5, 6 and 9.
  - What the frozen model did with them, seen only after the draw. Bed 6: first alert at hour 12, 31 hours before onset at hour 43, so very early. At the onset hour its score was back under the line (0.45 of it), though it was ranked 1st of 7. Bed 5: missed. At its onset hour, 48, it was ranked 3rd of 5 with a score of 0.14 of the line. Bed 9: first alert 282 hours before an onset at hour 337, very early (the record ends at hour 336). Of the 9 who never develop sepsis, 1 got a false alarm (bed 8, at hour 58).
  - So the board shows no timely catch: two very early alerts, one miss, one false alarm. That is what the draw gave, and it matches the hospital B result (9.5% timely, 33% very early, 57% missed). It stays. Twelve patients measure nothing, so none of this is reported as a result.
  - Also changed on the Replay: the score tile now leads with a plain word ("Below the line," "Near the line," "Over the line"). The exact number and "the model score divided by the frozen alert cutoff, not a probability" stay next to it. Any bed on the board can be opened in the Replay at the hour the board is showing.
  - Checks: 23 tests pass, 6 of them new for the board. One cuts each record off at an hour, scores it again, and confirms the board shows the same number, so the board cannot see the future. The app was checked by hand at desktop and phone width.
- Sun 11/15, 6:48 PM. I moved the Replay to be the first tab, so the app opens on the tile screen. The Unit board is now the second tab. Only the order of the two tabs changed: no model, threshold, patient or reported number changed. Checks: 23 tests pass, and in the running app the page opens on the Replay, Step moves the hour, and a bed button on the Unit board still opens that bed in the Replay.

## Sunday night: the alert cutoff and a different library (after-the-fact)

- Sun 11/15, 6:21 PM. PLAN, written before running. The goal tonight: a higher utility score. Two tests, both after-the-fact, both on the both-hospitals model from Saturday (the best model so far: utility 0.402 at hospital A and 0.406 at hospital B at the 25% cap). The frozen version 1 files are not read or written. No new data.
  - Test 1, the alert cutoff (scripts/v2_nested_scores.py saves the scores, scripts/v2_cutoff_policy.py applies the rules). The model is not changed. Saturday's run is repeated with the same patients, folds, seed, 100 inputs, settings and 200 trees, and this time the inner-fold scores are saved, so a cutoff rule can be chosen on patients the model did not train on and then scored on other patients. The first check is that the 25% cap reproduces Saturday's numbers exactly. Then three rules, each with every hospital's own cutoff:
    - (a) Higher caps: 30%, 35%, 40% and 50%. This is a trade, not an improvement: more sepsis patients alerted, more false alarms. Saturday's scouting curve on hospital A put the top of the utility curve near a 40% cap (0.422).
    - (b) The cutoff that maximizes utility, with no cap. For each hospital, caps from 5% to 60% in steps of 2.5 points are tried on the inner-fold patients, and the one with the highest official utility there is applied to the outer-fold patients. This is how the challenge teams set their cutoffs, so it is the number to put next to their scores.
    - (c) Cutoffs by time in the unit, at the 25% cap. Three bands: hours 1 to 12, 13 to 48, and 49 on. The early and late bands use the middle band's cutoff times a factor from 0.5, 0.7, 1, 1.4 and 2. For each of the 25 pairs the cutoff is set so at most 25% of inner-fold patients without sepsis are alerted, and the pair with the highest inner-fold utility is applied to the outer fold. Source of the idea: the outside literature review (stage of stay matters; one team's time-phased model).
  - Test 2, a different library (the same two scripts, run with --library lgbm). LightGBM 4.6.0 in place of scikit-learn's gradient boosting. Same patients, folds, 100 inputs, 25% cap and nesting. Settings fixed now to mirror the ones in use: learning rate 0.03, 31 leaves, at least 200 rows per leaf, L2 of 10. Two arms from the same fit, 200 trees and 400 trees. Install note: the LightGBM file was pointed at the OpenMP library that scikit-learn already ships, because the Mac has none installed. No other package changed (pip list compared before and after).
  - Rule for keeping (c) or LightGBM, set now: utility at least 0.01 higher at hospital A and at hospital B than Saturday's both-hospitals numbers at the 25% cap, and alerted by onset not more than 1.5 points lower at either. Saturday's keep rule (1.5 points more alerted by onset, utility not more than 0.01 lower) is reported next to it. Rules (a) and (b) have no keep rule: they change the operating point, and the cap is my call. The 25% cap stays the version 2 setting unless I change it.
  - Leak check, set now: any utility above 0.45 is treated as a possible mistake and checked before it is reported.
  - The guesses, logged in advance: (b) lands near a 40% to 45% cap with utility 0.42 to 0.44 at both hospitals and about 40% false alarms. (c) moves utility by 0.00 to plus 0.01 and does not meet the rule. LightGBM lands within 0.005 of scikit-learn at both tree counts and does not meet the rule. About a one in six chance that (c) or LightGBM passes.
- Sun 11/15, 6:40 PM. RESULT of both tests (results/after_the_fact_v2_cutoff_policy_hgb.csv and _lgbm.csv, with every chosen cap and factor in the .json files). After-the-fact. Both-hospitals model, nested, each hospital's own cutoff. Checks that passed: the repeat run's scores match Saturday's saved scores exactly (largest gap 0.0), the 25% cap reproduces Saturday's result exactly, and the fast copy of the utility used to choose rules matches the organizers' function. Every number below is from the organizers' function. Nothing is above the 0.45 leak line. 23 tests pass.

  | Rule (scikit-learn model) | A utility | A timely | A by onset | A missed | A false alarms | B utility | B timely | B by onset | B missed | B false alarms |
  |---|---|---|---|---|---|---|---|---|---|---|
  | 25% cap (Saturday's version 2) | 0.402 | 295 of 1,432 | 72.1% | 399 | 3,766 of 14,836 (25.4%) | 0.406 | 364 of 1,142 | 80.7% | 220 | 4,755 of 18,858 (25.2%) |
  | 30% cap | 0.412 | 316 | 76.7% | 333 | 4,514 (30.4%) | 0.404 | 362 | 84.7% | 175 | 5,915 (31.4%) |
  | 35% cap | 0.420 | 344 | 81.8% | 261 | 5,274 (35.5%) | 0.396 | 357 | 87.3% | 145 | 6,791 (36.0%) |
  | 40% cap | 0.422 | 359 | 85.3% | 210 | 5,996 (40.4%) | 0.387 | 363 | 89.3% | 122 | 7,661 (40.6%) |
  | 50% cap | 0.415 | 367 | 89.9% | 145 | 7,444 (50.2%) | 0.352 | 368 | 93.8% | 71 | 9,660 (51.2%) |
  | (b) Cutoff that maximizes utility, no cap | 0.422 | 365 | 88.1% | 170 | 6,737 (45.4%) | 0.407 | 359 | 80.8% | 219 | 4,896 (26.0%) |
  | (c) Time bands at the 25% cap | 0.396 | 281 | 69.6% | 436 | 3,716 (25.0%) | 0.402 | 360 | 80.5% | 223 | 4,776 (25.3%) |

  | LightGBM at the 25% cap | A utility | A timely | A by onset | A missed | B utility | B timely | B by onset | B missed |
  |---|---|---|---|---|---|---|---|---|
  | 200 trees | 0.400 | 312 | 72.6% | 392 | 0.406 | 360 | 80.5% | 223 |
  | 400 trees | 0.397 | 304 | 72.9% | 388 | 0.406 | 352 | 80.6% | 221 |

  - (b) The caps chosen on the inner folds were 47.5%, 45% and 42.5% at hospital A and 27.5%, 25% and 25% at hospital B. So at hospital B the 25% cap is already at the top of the utility curve, and the utility-maximizing cutoff changes nothing there (0.406 to 0.407). At hospital A it moves utility from 0.402 to 0.422 and missed patients from 399 to 170, and the price is false alarms on 45.4% of patients without sepsis in place of 25.4%: 5.3 false alarms per alerted sepsis patient and 652 alerts per 1,000 patient-days against 366.
  - (c) Time bands: worse at both hospitals (0.396 and 0.402). Rule not met. The factors chosen on the inner folds did not hold up on the outer folds.
  - LightGBM: a tie. Utility is 0.002 and 0.005 lower at hospital A and the same at hospital B. Tonight's rule is not met and Saturday's keep rule is not met (alerted by onset moved by plus 0.5 and plus 0.8 points at A, minus 0.2 and minus 0.1 at B). With its own utility-maximizing cutoff LightGBM reaches 0.420 at A and 0.406 to 0.408 at B, the same as scikit-learn. LightGBM is not in requirements.txt and nothing in the notebook or app uses it.
  - Guess against result. (b): right at hospital A (guessed 0.42 to 0.44 near a 40% to 45% cap, got 0.422 at about 45%), wrong at hospital B (got 0.407 at about 26%, because B's curve peaks much lower than A's). (c): guessed 0.00 to plus 0.01, got minus 0.006 and minus 0.004. LightGBM: guessed within 0.005, got that. Nothing passed, as guessed.
  - What it means: two different libraries, two tree counts, and three ways of setting the cutoff all land at 0.40 to 0.42. That is the ceiling of what these inputs can rank, not a setting that is off. The one real choice is the cap at hospital A: 0.422 is available there for anyone willing to alert on 45% of patients who never develop sepsis. Hospital B has nothing to gain from a higher cap. The version 2 setting stays the 25% cap unless I change it.

## Why no new outside data

The other open intensive care datasets either need a credentialed application that takes days, or have no ready-made hourly sepsis label. Building that label ourselves is days of careful work and easy to get wrong. A third hospital is listed as future work.

## Result vs prediction

The one hospital B run finished at 3:13 PM on Sat 11/14 (commit deb692a), using the models and thresholds frozen at 3:10 PM (commit c3fbed6). Hospital B: 20,000 patients, 1,142 with sepsis.

| Prediction | What happened | Verdict |
|---|---|---|
| 1. Utility goes down, by a third to a half | 0.368 at A dev to 0.238 at B (95% range 0.214 to 0.262). A drop of 35% | Right |
| 2. False alarms go above the 15% budget (low confidence) | 1,358 of 18,858, which is 7.2%. Half the budget, not more | Wrong |
| 3. M3 loses more than M1 when it travels | M3 utility 0.368 to 0.238 (down 35%). M1 0.197 to 0.091 (down 54%). Alerted by onset: M3 62% to 43%, M1 53% to 28% | Mostly wrong. M3 lost a little more utility in absolute terms (0.130 against 0.106), but M1 lost more of what it had and more catches |

What I did not predict:
- The model got quieter at hospital B, not noisier. Scores ran lower there, so the frozen alert line was crossed less often, for sepsis patients as well. Catches by onset fell from 222 of 358 (62%) to 486 of 1,142 (43%).
- Time alone matched the full model. T0 scored utility 0.242 at B and M3 scored 0.238.
- Alerting on the same share of all patients as at A dev (a label-free adjustment) brought catches by onset back to 55% (629 of 1,142) at a 17.1% false alarm rate, utility 0.270.
- Late-entry cases: M3 caught 14 of 223. The modified SIRS rule caught 150 of 223.
- The modified SIRS rule at B: alerted by onset on 966 of 1,142, false alarms on 12,630 of 18,858 (67%), utility -0.014.

Why prediction 2 was wrong (written Sat 11/14, 4:35 PM). What was measured, without using sepsis labels: the frozen model gave hospital B patients lower scores across the board, so the frozen alert line was crossed less often, by patients without sepsis (7.2% against 15.0%) and by sepsis patients as well (43% alerted by onset against 62%). Two differences in the recorded data go with that: hospital B records 0.94 lab values per patient-hour against 1.64 at hospital A, and its blood pressures read higher (mean arterial pressure 86.5 against 78.6). These are associations. I did not run a test that proves either one is the cause. The prediction assumed ordinary hospital B patients might look sicker to a model raised on hospital A. They looked healthier.
