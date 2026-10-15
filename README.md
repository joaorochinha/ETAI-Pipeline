# ETAI Predictive Pipeline — COMPAS recidivism

**João Rochinha** · student no. 20260600 · NOVA IMS, Exploratory Topics in Artificial Intelligence

Fork of the course's baseline pipeline, extended every week.

## Overview

The pipeline predicts `two_year_recid` (was the person rearrested within two years?) from ProPublica's COMPAS data. `race` is never used as a feature: it is kept aside only to check whether the model makes more mistakes for some groups than for others. COMPAS's own score (`score_text`) is also kept aside, to compare our model with it.

**Current best result:** the three tuned ensembles are tied at **0.681–0.683 nested CV accuracy** (AdaBoost 0.683, Oblique Random Forest 0.682, Random Forest 0.681). The locked test set has not been used yet.

## Project structure

```text
main.py              entry point: python main.py
config.yaml          all settings (cleaning rules, preprocessing, model, CV, tuning)
requirements.txt     package versions
src/data.py          loading
src/preprocessing.py cleaning, duplicates, preprocessing (ColumnTransformer), locked test split
src/model.py         models (incl. oblique random forest and AdaBoost) + build_pipeline()
src/evaluate.py      cross-validation, classification report, fairness report
src/tuning.py        Optuna tuning + nested cross-validation
src/results.py       saves each run (report + copy of the config) to results/
data/                dataset + data dictionary
```

## Pipeline progress

| Week | Focus | What changed |
|---|---|---|
| 2 | Baseline | One train/test split, rows with missing values dropped, Logistic Regression vs Decision Tree, first fairness check (FPR by race). |
| 3 | EDA + preprocessing | Missing-value analysis, invalid values → NaN, placeholders → NaN, category spelling fixed, duplicates removed, redundant columns dropped, imputation + `_was_missing` flags. |
| 4 | Cross-validation | Locked test set (20%), stratified 5-fold CV of the whole pipeline, out-of-fold reports, Dummy and Random Forest added. |
| 5 | Hyperparameter tuning | Optuna tuning on the development set, nested CV for an honest score. Tuned Decision Tree, Logistic Regression and Random Forest. |
| 5+ | Challenges | Week 3: consistency rules (`age_cat` vs `age`, `score_text` vs `decile_score`). Week 4: KNN imputation tested. Week 5: tuning the imputation strategy, tuning for balanced accuracy. Same number of trials (30) for every model. Balanced accuracy added to the report. Each run saves a copy of its config. |
| 6 | Ensembles | New models in `src/model.py`: `oblique_random_forest()` (random projections + trees, bagged) and `adaboost()` (boosted trees). New search spaces for random forest, oblique random forest and AdaBoost; the random forest now has `n_estimators: 200` fixed. Challenge 1: `add_engineered_features()` with three new features, tested with nested CV. |

## Preprocessing decisions

| Column(s) | Problem | What was done |
|---|---|---|
| `age`, `juv_fel_count`, `sex` | some missing values, at random (MCAR) | median / most frequent value |
| `priors_count`, `c_charge_degree` | missing values linked to `age_cat`, so not completely at random | median / most frequent value + a `_was_missing` flag |
| `race` | some missing values | not imputed (not a feature); shown as `unknown` in the fairness report |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | impossible values (e.g. age −3, decile 23, priors 500) | set to NaN |
| `priors_count`, `race`, `sex` | placeholders such as `-` and `?` | set to NaN |
| `sex`, `race`, `c_charge_degree`, `score_text` | same category written in different ways (`male`, `MALE`…) | one spelling |
| `age_cat` vs `age` | 6 people aged 51–66 recorded as "Less than 25" | `age_cat` recalculated from `age` |
| `score_text` vs `decile_score` | 10 rows where the text does not match the decile | `score_text` recalculated from `decile_score` |
| whole rows | 72 duplicates | removed before the test split |
| `prior_offenses`, `age_in_months`, `juvenile_total` | copies of other columns | dropped |

After cleaning: **5771 rows for development** and **1443 rows in the locked test set**.

Encoder and scaler: target encoding (one column per categorical feature) and robust scaling (less affected by extreme counts).

### Week 4 challenge: KNN imputation

Same 5 folds for every option, default hyperparameters. Holdout = one 75/25 split of the development set.

| Imputation | LR holdout | LR CV | DT holdout | DT CV | RF holdout | RF CV |
|---|---:|---:|---:|---:|---:|---:|
| **Median** | 0.674 | 0.672 ± 0.013 | 0.595 | 0.614 ± 0.012 | 0.641 | 0.646 ± 0.016 |
| KNN (k = 5) | 0.672 | 0.672 ± 0.012 | 0.595 | 0.605 ± 0.010 | 0.640 | 0.643 ± 0.019 |
| KNN (k = 15) | 0.672 | 0.673 ± 0.014 | 0.607 | 0.607 ± 0.014 | 0.637 | 0.644 ± 0.020 |

For KNN the data is scaled before imputing, because KNN uses distances. The differences between the options are much smaller than the standard deviation between folds, so KNN does not help. The single holdout cannot tell the options apart either: its numbers move as much as its own noise. **Median stays** because it is simpler.

## Model evaluation (week 4)

| Model | Holdout W2 | Holdout W3 | CV accuracy (mean ± std) | Train − validation gap |
|---|---:|---:|---:|---:|
| Dummy | — | — | 0.549 ± 0.000 | 0.000 |
| Logistic Regression | 0.677 | 0.658 | **0.672 ± 0.013** | +0.004 |
| Decision Tree | 0.626 | 0.602 | 0.614 ± 0.012 | +0.082 |
| Random Forest | — | — | 0.646 ± 0.016 | +0.086 |

*W2/W3 holdout values come from my earlier `results/` files. W2 used a smaller test set (rows with missing values were dropped).*

The CV number is more reliable than a single holdout, because every development row is used for validation and we also get the standard deviation. With default settings, Logistic Regression is the best model. The Decision Tree and Random Forest overfit: training accuracy is about 0.08 higher than validation accuracy.

## Hyperparameter tuning (week 5)

Optuna, **30 trials per model**. Each trial is scored by 5-fold CV of the whole pipeline on the development set. Nested CV (5 outer folds) gives the honest score. The test set is not used.

| Model | Default CV | Tuned (nested CV) | Best trial | Optimism | Chosen hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision Tree | 0.614 ± 0.012 | 0.676 ± 0.019 | 0.680 | −0.001 | `max_depth=6`, `min_samples_leaf=110`, `criterion=entropy` |
| Logistic Regression | 0.672 ± 0.013 | 0.673 ± 0.015 | 0.678 | +0.002 | `C=0.079` |
| **Random Forest** | 0.646 ± 0.016 | **0.682 ± 0.017** | 0.683 | +0.001 | `n_estimators=191`, `max_depth=10`, `min_samples_leaf=50`, `max_features=log2` |

- **The Decision Tree gained the most (+0.062)**, much more than its standard deviation. Limiting the depth and requiring at least 110 people per leaf stopped the overfitting: the gap went from 0.082 to 0.009.
- **The Random Forest also improved (+0.036)**, for the same reason (`min_samples_leaf=50`).
- **Logistic Regression barely changed (+0.001).** It was not overfitting before, so there was little to fix.
- **Tuning changed the best model:** the Random Forest now has the highest score. But the difference with the other models (≤ 0.009) is smaller than the standard deviation, so it is not clearly better.
- **The number to report is the nested CV score (0.682)**, not the best trial (0.683). The best trial is the maximum of many noisy scores, so it is optimistic.

| Tuned model | Accuracy | Balanced accuracy | Recall of class 1 (reoffended) |
|---|---:|---:|---:|
| Decision Tree | 0.68 | 0.665 | 0.55 |
| Logistic Regression | 0.67 | 0.657 | 0.50 |
| Random Forest | 0.68 | 0.670 | 0.55 |

All models miss about half of the people who reoffend (recall 0.50–0.55). Accuracy alone hides this.

### Week 5 challenges 3 and 4

**3. Tuning the imputation too** (median or mean, added to the Decision Tree's search space). The nested CV was 0.672 ± 0.015, about the same as without it (0.676). The final search kept the median. The imputation strategy makes no difference here. It is still leak-free, because the imputer is inside the pipeline and is fitted again on the training part of every fold.

**4. Tuning for balanced accuracy** instead of accuracy:

| Model | Balanced accuracy / recall (tuned for accuracy) | Balanced accuracy / recall (tuned for balanced accuracy) |
|---|---|---|
| Decision Tree | 0.665 / 0.55 | 0.663 / 0.58 |
| Logistic Regression | 0.657 / 0.50 | 0.657 / 0.50 |
| Random Forest | 0.670 / 0.55 | 0.671 / 0.57 |

Very small changes. The model still predicts "reoffends" only when the probability is above 0.5, so changing the metric alone does not change much. Accuracy stays as the tuning metric.

## Ensembles (week 6)

Three ensembles, each tuned with the week 5 machinery (Optuna, 30 trials, nested CV on the same 5 outer folds). Logistic Regression and the Decision Tree are repeated so everything is in one table.

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Best trial | Chosen hyperparameters |
|---|---:|---:|---:|---|
| Logistic Regression | 0.672 ± 0.013 (+0.004) | 0.673 ± 0.015 (+0.004) | 0.678 | `C=0.079` |
| Decision Tree | 0.614 ± 0.012 (+0.082) | 0.676 ± 0.019 (+0.009) | 0.680 | `max_depth=6`, `min_samples_leaf=110`, `criterion=entropy` |
| Random Forest | 0.647 ± 0.016 (+0.085) | 0.681 ± 0.015 (+0.008) | 0.684 | `max_depth=24`, `min_samples_leaf=45`, `max_features=0.5` |
| Oblique Random Forest | 0.648 ± 0.014 (+0.114) | 0.682 ± 0.014 (+0.009) | 0.683 | tree `max_depth=4`, `min_samples_leaf=3`, `n_components=8` |
| **AdaBoost** | 0.676 ± 0.013 (+0.006) | **0.683 ± 0.014** (+0.005) | 0.683 | base tree `max_depth=4`, `n_estimators=220`, `learning_rate=0.025` |

*Default params: forests with 200 trees, AdaBoost with 200 stumps and `learning_rate=1.0`. The random forest uses this week's search space (`n_estimators` fixed at 200), so its numbers differ slightly from the week 5 table.*

| Tuned model | Balanced accuracy | Recall of class 1 |
|---|---:|---:|
| Random Forest | 0.671 | 0.57 |
| Oblique Random Forest | 0.670 | 0.55 |
| AdaBoost | 0.672 | 0.56 |

**What the table shows.**
- **Tuning helped most where the defaults overfit.** Both forests grow full trees by default (gaps +0.085 and +0.114). Tuning lifts them by about 3.5 points and closes the gap. AdaBoost with stumps was already well behaved (gap +0.006), so tuning added only a little (+0.007).
- **The three tuned ensembles are tied.** AdaBoost has the best mean, but only +0.002 over the random forest and +0.001 over the oblique forest, with folds going both ways (it is better in 2 of 5 folds against the random forest). These differences are much smaller than the fold std (~0.015), so choosing a winner among them would be choosing noise.
- **Against the week 5 models, the ensembles gain about one point.** AdaBoost is +0.006 over the tuned tree and +0.010 over logistic regression on average (better in 4 of 5 folds), still smaller than the fold std.
- **The hyperparameters match how each model works.** The random forest keeps deep trees (`max_depth=24`) but needs at least 45 people per leaf, so trees only grow where there is enough data. AdaBoost chose weak base trees (depth 4) with a small learning rate (0.025) and many rounds (220), which is the trade-off seen in the notebook. The oblique forest chose shallow trees (depth 4).
- **The oblique forest does not beat the normal one.** The signal in COMPAS is mostly "more priors, younger", which normal one-feature splits already express well.
- **Every model ends near 0.68.** Five very different models hitting the same limit suggests the limit is the information in these 10 features, not the model.

**Which number to report:** the nested CV score of the chosen model, e.g. **0.683 ± 0.014** for AdaBoost, not the best trial.

### Week 6 challenge 1: feature engineering

`add_engineered_features()` (in `src/preprocessing.py`, called in `main.py` before `split_features_target()`) adds three columns. They use fixed formulas only, so nothing is learned from the data and it is safe before the split:

| Feature | Formula | Idea |
|---|---|---|
| `priors_per_adult_year` | `priors_count / (age − 17)` | 2 priors at 19 is not the same as 2 priors at 45 (the notebook's example) |
| `has_juvenile_record` | 1 if any juvenile offence, else 0 | having a juvenile record at all |
| `young_felony` | 1 if `age_cat` is "Less than 25" **and** the charge is a felony | interaction between age and charge |

Nested CV with and without the three features (same folds, same search spaces):

| Model | Without features | With features | Difference | Folds where it helped |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.673 ± 0.015 | 0.670 ± 0.011 | −0.003 | 2 of 5 |
| Random Forest | 0.681 ± 0.015 | 0.673 ± 0.009 | −0.008 | 1 of 5 |
| AdaBoost | 0.683 ± 0.014 | 0.676 ± 0.009 | −0.007 | 1 of 5 |

**No model family gains.** With default hyperparameters Logistic Regression went from 0.672 to 0.676, but after tuning the gain disappears. The trees can already combine `age` and `priors_count` themselves. So the features are computed but **not used**: they are not listed in `preprocessing.numeric_features` (adding them there switches them on).

*Challenge 2 (AdaBoost over Logistic Regression) was not done. Boosting adds up linear models, and a weighted sum of linear models is still a linear model, so no big gain is expected.*

## Fairness check

False positive rate (FPR) = share of people who did **not** reoffend but were predicted to. Tuned Random Forest (week 6), out-of-fold predictions on the development set:

| Group | Our model FPR | COMPAS FPR | People who did not reoffend |
|---|---:|---:|---:|
| African-American | 0.30 | 0.45 | 1420 |
| Caucasian | 0.17 | 0.23 | 1161 |
| Hispanic | 0.17 | 0.23 | 311 |
| Other | 0.17 | 0.14 | 185 |
| unknown | 0.20 | 0.33 | 66 |
| Asian | 0.14 | 0.10 | 21 |
| Native American | 0.17 | 0.17 | 6 |

- Our model has a lower FPR than COMPAS for the largest groups. Part of the reason is that it predicts "reoffends" less often (recall 0.57).
- **The gap between groups is still there.** African-American people who did not reoffend are wrongly flagged almost **twice** as often as Caucasian people, both by our model (0.30 vs 0.17) and by COMPAS (0.45 vs 0.23). AdaBoost and the oblique forest give almost the same rates (0.29 vs 0.15–0.16).
- `race` is not a feature, but other features (for example `priors_count`) are related to it, so removing `race` does not remove the difference.
- Asian and Native American groups are too small for reliable rates.

## Current best model

The three tuned ensembles are tied (0.681–0.683). `config.yaml` runs the **Random Forest** (the course default this week): `n_estimators=200`, `max_depth=24`, `min_samples_leaf=45`, `max_features=0.5`, refitted on all 5771 development rows, nested CV accuracy **0.681 ± 0.015**. AdaBoost has the highest mean (0.683) but the difference is noise. Logistic Regression (0.673) is only one point lower and much easier to interpret.
