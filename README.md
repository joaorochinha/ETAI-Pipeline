# ETAI Predictive Pipeline — COMPAS recidivism

**João Rochinha** · student no. 20260600 · NOVA IMS, Exploratory Topics in Artificial Intelligence

Fork of the course's baseline pipeline, extended every week.

## Overview

The pipeline predicts `two_year_recid` (was the person rearrested within two years?) from ProPublica's COMPAS data. `race` is never used as a feature: it is kept aside only to check whether the model makes more mistakes for some groups than for others. COMPAS's own score (`score_text`) is also kept aside, to compare our model with it.

**Current best result (selection metric ROC-AUC):** the tuned ensembles are tied at **nested CV ROC-AUC 0.726–0.728** (HistGradientBoosting, Gradient Boosting, the oblique forest and XGBoost all reach 0.728), so there is no clear winner. The locked test set has not been used yet.

## Project structure

```text
main.py              entry point: python main.py
config.yaml          all settings (cleaning rules, preprocessing, model, CV, tuning)
requirements.txt     package versions
src/data.py          loading
src/preprocessing.py cleaning, duplicates, preprocessing (ColumnTransformer), locked test split
src/model.py         models (incl. oblique random forest, AdaBoost, XGBoost), VoteSelector (feature selection) + build_pipeline()
src/evaluate.py      scorers for 8 metrics, cross-validation, classification report, fairness report, corrected paired t-test
src/tuning.py        Optuna tuning + nested cross-validation
src/results.py       saves each run (report + copy of the config) and adds one row to the scoreboard (results/runs.csv)
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
| 7 | Boosting + metrics | New models: `gradient_boosting`, `hist_gradient_boosting`, `naive_bayes`, with search spaces. Every run now reports **8 metrics** from out-of-fold **probabilities**; tuning maximises one **selection metric** (`roc_auc`). Every run adds a row to the **scoreboard** (`results/runs.csv`) with the no-model floor and the untuned score. `class_weight: balanced` for the models that support it. `corrected_paired_test()` added to `src/evaluate.py`. |
| 8 | XGBoost + feature selection | New model `xgboost` with its search space (incl. `gamma`, `reg_alpha`). **`VoteSelector`**: a feature-selection step **inside** the Pipeline (between preprocessing and model), so it is re-fitted on the training rows of every fold. `select__min_votes` is tuned together with the model. Fitted steps are cached (`memory=`) during each run. The kept features are printed, and the scoreboard records the selection setting. |

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

## Metrics and scoreboard (week 7)

From week 7 on, every run reports 8 metrics (accuracy, balanced accuracy, precision, recall, F1, F-beta 2, ROC-AUC, average precision), all from out-of-fold **probabilities**. Tuning maximises **one** of them, the selection metric. The tables above (weeks 4–6) use accuracy, so their numbers are not comparable with the ones below.

**Selection metric: ROC-AUC.** COMPAS is a risk score that is used to **rank** people, and ROC-AUC measures exactly that: the chance that a random person who reoffended gets a higher score than a random person who did not. It does not depend on the 0.5 cut-off, so choosing the threshold stays a separate decision. It is also less noisy than accuracy, because it uses the whole ranking and not only which side of 0.5 each person falls on.

### Scoreboard

Every model, tuned with nested CV (30 Optuna trials, 5 outer × 5 inner folds, same folds for every model). *Untuned* = the same model with its `config.yaml` params; *floor* = a dummy model that predicts the same for everyone.

| Model | Untuned ROC-AUC | **Tuned ROC-AUC (nested CV)** | Floor | Chosen hyperparameters |
|---|---:|---:|---:|---|
| **HistGradientBoosting** | 0.720 | **0.728 ± 0.015** | 0.500 | `max_iter=402`, `learning_rate=0.039`, `max_leaf_nodes=3`, `min_samples_leaf=117`, `l2_regularization=0.0002` |
| Gradient Boosting | 0.728 | 0.728 ± 0.016 | 0.500 | `n_estimators=190`, `learning_rate=0.016`, `max_depth=4`, `min_samples_leaf=76`, `subsample=0.60` |
| Oblique Random Forest | 0.686 | 0.728 ± 0.013 | 0.500 | tree `max_depth=29`, `min_samples_leaf=87`, `n_components=8` |
| AdaBoost | 0.723 | 0.727 ± 0.016 | 0.500 | base tree `max_depth=4`, `n_estimators=55`, `learning_rate=0.167` |
| Random Forest | 0.683 | 0.726 ± 0.014 | 0.500 | `max_depth=7`, `min_samples_leaf=3`, `max_features=sqrt` |
| Logistic Regression | 0.719 | 0.718 ± 0.014 | 0.500 | `C=0.0022` |
| Decision Tree | 0.606 | 0.718 ± 0.016 | 0.500 | `max_depth=6`, `min_samples_leaf=104`, `criterion=gini` |
| Naive Bayes | 0.685 | 0.715 ± 0.011 | 0.500 | `var_smoothing=0.90` |
| Dummy (floor) | 0.500 | — | 0.500 | — |

*Params: Logistic Regression, Decision Tree, Random Forest and HistGradientBoosting use `class_weight: balanced` (the commented examples in `config.yaml`). Gradient Boosting, AdaBoost, the oblique forest and Naive Bayes have no `class_weight`.*

The other metrics of the tuned models (nested CV means, threshold 0.5):

| Model | Accuracy | Balanced accuracy | Precision | Recall | Average precision |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 0.677 | 0.674 | 0.643 | 0.637 | 0.694 |
| Gradient Boosting | 0.681 | 0.671 | 0.675 | 0.564 | 0.693 |
| Oblique Random Forest | 0.680 | 0.670 | 0.672 | 0.569 | 0.686 |
| AdaBoost | 0.686 | 0.675 | 0.683 | 0.565 | 0.689 |
| Random Forest | 0.674 | 0.670 | 0.641 | 0.632 | 0.689 |
| Logistic Regression | 0.677 | 0.668 | 0.662 | 0.583 | 0.671 |
| Decision Tree | 0.668 | 0.664 | 0.636 | 0.621 | 0.670 |
| Naive Bayes | 0.604 | 0.566 | 0.757 | 0.179 | 0.664 |

**Best model against each of the others** (corrected paired t-test on the 5 outer folds):

| HistGradientBoosting vs | Difference in ROC-AUC | Folds where it wins | p-value |
|---|---:|---:|---:|
| Gradient Boosting | +0.000 | 3/5 | 0.80 |
| Oblique Random Forest | +0.001 | 3/5 | 0.67 |
| AdaBoost | +0.001 | 3/5 | 0.55 |
| Random Forest | +0.003 | 4/5 | 0.20 |
| Logistic Regression | +0.010 | 5/5 | 0.005 |
| Decision Tree | +0.011 | 5/5 | 0.015 |
| Naive Bayes | +0.013 | 5/5 | 0.02 |

**What the scoreboard shows.**
- **Every model is well above the floor (0.5)**, and the ensembles rank people better than the simple models: about 0.73 against 0.72 for Logistic Regression, the tree and Naive Bayes.
- **The top five are tied.** HistGradientBoosting has the best mean by a hair, but against Gradient Boosting, the oblique forest, AdaBoost and the random forest the differences are 0.000–0.003 (p ≥ 0.20). Against Logistic Regression (+0.010, p = 0.005), the tree (+0.011, p = 0.015) and Naive Bayes (+0.013, p = 0.02) the difference is clear.
- **Tuning helped most where the defaults overfit:** Decision Tree +0.112, oblique forest +0.042, Random Forest +0.043. HistGradientBoosting (+0.008) and AdaBoost (+0.004) gained a little. **Gradient Boosting and Logistic Regression did not gain** (0.728 → 0.728, 0.719 → 0.718): their defaults were already good.
- **Naive Bayes shows why the selection metric matters.** It ranks people almost as well as Logistic Regression (ROC-AUC 0.715), but at the 0.5 threshold it flags very few people (recall 0.18), so its accuracy is only 0.604.
- **`class_weight: balanced` trades precision for recall.** The tree-based models that use it (HistGradientBoosting, Random Forest, Decision Tree) have recall 0.62–0.64, against 0.56–0.57 for Gradient Boosting, AdaBoost and the oblique forest, which don't use it (Logistic Regression, also balanced: 0.58).

*Part B (feature selection with `VoteSelector`) is optional and was not done.*

## Feature selection and XGBoost (week 8)

**XGBoost** is gradient boosting with extra brakes inside each tree (`gamma`: minimum gain to split, `reg_alpha` / `reg_lambda`: L1 / L2 penalties on the leaves, `min_child_weight`). Its defaults are aggressive (depth 6, learning rate 0.3), so it is always tuned.

**Feature selection** uses the `VoteSelector`: a random forest and gradient boosting vote for the features with above-average importance, and an L1 logistic regression votes for the features it does not shrink to zero. A feature is kept if it gets at least `min_votes` of the 3 votes. Selection looks at the labels, so it is a step **inside** the Pipeline and is learned again on the training rows of every fold. Selecting on all rows first and then cross-validating would leak: in the notebook, this made pure noise look like ROC-AUC 0.84.

### Does selection pay off?

XGBoost tuned twice with nested CV (30 trials, same folds), once with and once without selection. With selection on, `select__min_votes` (1–3) is tuned together with the model:

| XGBoost (tuned) | Untuned ROC-AUC | **Tuned ROC-AUC (nested CV)** | Features used |
|---|---:|---:|---:|
| Without selection | 0.725 | **0.728 ± 0.015** | 10 |
| With selection (`VoteSelector`) | 0.721 | **0.726 ± 0.013** | 3 |

**Corrected paired t-test (with − without):** −0.002, selection better in 1 of 5 folds, **p = 0.42**.

**Features kept** by the final model: `age` and `priors_count` (both 3/3 votes), plus `age_cat` when `min_votes = 1` (it gets 1 vote). Every other feature got 0 votes (the juvenile counts, `sex`, `c_charge_degree` and the two missing-value flags). On my laptop the final search chose `min_votes = 2` and kept only `age` and `priors_count`; in the run used for the tables above it chose `min_votes = 1` and also kept `age_cat`.

*Tuning results can differ slightly between machines.* XGBoost runs on several threads, so scores can change in the 4th decimal. Optuna picks each next trial from the earlier scores, so a tiny change can lead the search to different hyperparameters. The honest nested CV estimate is the same on both machines (0.721 untuned, **0.726 ± 0.013** tuned).

**Is the smaller model as good?** Yes. With 3 features instead of 10, ROC-AUC drops by only 0.002, much less than the fold std (~0.013–0.015), and the corrected test finds no real difference (p = 0.42). The other metrics are almost identical too (accuracy 0.679 vs 0.682, recall 0.57 in both), and so is the fairness check (FPR 0.31 vs 0.17 with selection, 0.30 vs 0.16 without).

**Would I keep selection on?** Yes. It gives the same performance with a model that depends on only age and number of priors, which is easier to explain and to audit. The cost is a slower run (three extra models per fold), which the cache keeps small. But selection does not *improve* the score here: COMPAS has only 10 features and almost no noise to remove, unlike the notebook's example with 20 random columns.

**Week 8 B2: how many votes?** Nested CV chose `min_votes = 1` (the least strict option) in **4 of the 5 outer folds** (fold 1 chose 2) and in the final search. The tuning mostly prefers keeping as many features as possible. Even with 1 vote, only 3 features get any vote at all, so the kept set barely changes: `min_votes = 2` would drop only `age_cat`. *(B1, other selectors, was not done.)*

**XGBoost against week 7:** tuned XGBoost without selection (0.728) is in the same group as the week 7 ensembles (same folds). It does not change the ranking.

## Fairness check

False positive rate (FPR) = share of people who did **not** reoffend but were predicted to (threshold 0.5). Tuned XGBoost with feature selection (the model in `config.yaml`), out-of-fold predictions on the development set:

| Group | Our model FPR | COMPAS FPR | People who did not reoffend |
|---|---:|---:|---:|
| African-American | 0.31 | 0.45 | 1420 |
| Caucasian | 0.17 | 0.23 | 1161 |
| Hispanic | 0.17 | 0.23 | 311 |
| Other | 0.18 | 0.14 | 185 |
| unknown | 0.20 | 0.33 | 66 |
| Asian | 0.10 | 0.10 | 21 |
| Native American | 0.17 | 0.17 | 6 |

- Our model has a lower FPR than COMPAS for the largest groups. Part of the reason is that it predicts "reoffends" less often (recall 0.57).
- **The gap between groups is still there.** African-American people who did not reoffend are wrongly flagged about **twice** as often as Caucasian people, both by our model (0.31 vs 0.17) and by COMPAS (0.45 vs 0.23).
- **Feature selection does not remove it either.** The model now uses only `age` and `priors_count` (plus `age_cat`), and the gap is the same. `race` is not a feature, but `priors_count` is related to it, so it carries the difference.
- In week 7, the balanced HistGradientBoosting flagged more people and had higher FPR in every group (0.37 vs 0.22): `class_weight: balanced` trades fewer missed reoffenders for more false alarms.
- Asian and Native American groups are too small for reliable rates.

## Current best model

On ROC-AUC the ensembles are tied (0.726–0.728). `config.yaml` runs **XGBoost with feature selection** (the course default this week), refitted on all 5771 development rows: nested CV ROC-AUC **0.726 ± 0.013**. Chosen hyperparameters on my laptop: `n_estimators=303`, `learning_rate=0.044`, `max_depth=3`, `min_child_weight=2.21`, `subsample=0.78`, `colsample_bytree=0.58`, `reg_lambda=1.25`, `reg_alpha=0.147`, `gamma=2.31`, `min_votes=2`, so the model uses only `age` and `priors_count`. XGBoost without selection and three week 7 ensembles reach 0.728, but the differences are not significant. Logistic Regression (0.718) ranks clearly worse, but is much easier to interpret.