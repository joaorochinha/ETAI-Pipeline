# Baseline Predictive Pipeline -- ETAI

This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS dataset -- the data behind a real 2016 investigation into a risk-assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks go on.

---

## Project structure

```text
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml              # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py              # loading
│   ├── preprocessing.py     # row-preserving cleaning, training-only
│   │                        # de-duplication, deployable preprocessing,
│   │                        # and the locked final test-set split
│   ├── model.py             # model construction + build_pipeline()
│   ├── evaluate.py          # stratified k-fold CV, OOF reports + fairness
│   ├── tuning.py            # Optuna tuning + nested cross-validation
│   └── results.py           # saves each run's report to disk
├── results/                 # generated run reports, not tracked in git
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

---

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it is a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|---|---|---|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split, minimal preprocessing, Logistic Regression baseline, Decision Tree comparison, a first fairness check comparing our model's and COMPAS's false-positive rate by race, train-vs-test accuracy reporting, and automatic result saving to `results/`. |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | Missingness-mechanism analysis, domain-rule invalid-value detection, duplicate checks, category cleanup, mechanism-matched imputation with `_was_missing` indicators, a deployable `ColumnTransformer`, redundant-column removal and configuration-driven preprocessing. |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** is set aside and no longer used for routine model comparison. Models are evaluated using **stratified 5-fold cross-validation** of the complete preprocessing + model pipeline on the development set. Classification and fairness reports use out-of-fold predictions. `clean_dataset()` becomes row-preserving, duplicate removal becomes a separate training-only operation, sklearn `TargetEncoder` uses internal cross-fitting, and Dummy Classifier and Random Forest are added as model options. |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** on the development set only. Every candidate is evaluated by cross-validation of the complete pipeline. **Nested cross-validation** provides the honest performance estimate of the tuning procedure, while tuning once on the full development set selects the final hyperparameters. `build_pipeline()` is introduced in `src/model.py`, tuning configuration is stored in `config.yaml`, and `optuna` is added to `requirements.txt`. The Week 5 challenges were also completed by tuning **Decision Tree, Logistic Regression and Random Forest** and comparing all three using nested CV. The locked test set remains untouched. |

---

## Preprocessing decisions

*(Written from the diagnosis in `Practical/W3/notebooks/01_eda_introduction.ipynb`; the preprocessing walkthrough is in `Practical/W4/notebooks/02_preprocessing.ipynb`.)*

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | ~2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | ~3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing, including placeholder tokens | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | ~3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | missing / placeholder values | MCAR | mode impute, no indicator; excluded from model features |
| `sex` | ~1.5% missing | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values | domain rule | converted to `NaN` before imputation |
| `sex`, `race`, `c_charge_degree`, `score_text` | inconsistent category spelling | data entry | canonicalized |
| whole rows | 72 exact duplicates / repeated IDs | data entry | dropped from labelled training data, keeping first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns | multicollinearity | dropped |

**Encoder/scaler pair:** chosen in `config.yaml`:

```yaml
preprocessing:
  encoder: "target"
  scaler: "robust"
```

Target encoding provides a compact representation of categorical variables, while Robust Scaling uses the median and interquartile range and is less influenced by extreme count values.

The preprocessing pipeline also creates:

```text
priors_count_was_missing
c_charge_degree_was_missing
```

before imputation, so information about the original missingness remains available to the model.

---

## Model evaluation

From Week 4 onward, model comparison is performed using the **development set only**.

The final test set is locked before any model comparison:

```yaml
test_set:
  size: 0.2
  random_state: 42
```

After cleaning and duplicate removal, the split contains:

```text
Development set: 5771 rows
Locked test set: 1443 rows
```

The locked test set is not used to fit, tune or select models.

Models are compared using stratified 5-fold cross-validation:

```yaml
cv:
  n_splits: 5
  shuffle: true
  random_state: 42
  scoring: "accuracy"
```

Because preprocessing is inside the sklearn `Pipeline`, imputation, encoding, scaling and model fitting are repeated independently inside every training fold. Validation observations therefore cannot influence fitted preprocessing parameters.

### Week 4 cross-validation results

| Model | CV accuracy (mean ± std) | Train-validation gap |
|---|---:|---:|
| Dummy Classifier | 0.549 ± 0.000 | 0.000 |
| **Logistic Regression** | **0.672 ± 0.013** | **+0.003** |
| Decision Tree | 0.610 ± 0.018 | +0.085 |
| Random Forest | 0.647 ± 0.016 | +0.085 |

Before hyperparameter tuning, Logistic Regression had the highest mean validation accuracy and the smallest train-validation gap among the predictive models.

The Decision Tree and Random Forest both achieved considerably higher training performance than validation performance, indicating more overfitting.

The Dummy Classifier acts as the majority-class baseline. Its accuracy of approximately `0.549` shows the minimum level that a useful predictive model should beat.

The locked test set is not used to decide which model is best.

---

## Hyperparameter tuning

*(Walkthrough: `Practical/W5/notebooks/04_hyperparameter_tuning.ipynb`.)*

Week 5 introduces hyperparameter tuning using **Optuna**.

Hyperparameters are modelling decisions, so they must be selected using only the development set. The locked test set remains untouched throughout tuning.

With:

```yaml
tuning:
  enabled: true
```

`python main.py` performs two additional stages.

### Nested cross-validation

For each of the 5 outer folds, a complete Optuna study is run using only the outer-training observations.

Each Optuna candidate is itself evaluated using stratified cross-validation of the **whole pipeline**.

The best candidate is refitted using the outer-training rows and then evaluated on the outer-validation fold.

The outer-validation observations never participate in the hyperparameter search.

Therefore, the mean score across the outer folds is the honest estimate of the complete:

```text
tune
→ choose hyperparameters
→ refit
→ predict unseen data
```

procedure.

### Tuning on the whole development set

After nested cross-validation, the same Optuna search is performed once using all development observations.

This second search determines the hyperparameters kept by the final model.

The resulting tuned pipeline is then refitted on all **5771 development rows**.

### Best-trial optimism

Optuna also reports the best mean CV score obtained during the search.

That score is useful for selecting hyperparameters, but it is the **maximum of several noisy CV estimates**, so it can be optimistic.

For reporting expected tuned-model performance, the nested-CV score is preferred.

---

### Search spaces

The Decision Tree was tuned with **30 Optuna trials** using:

```text
max_depth
min_samples_leaf
criterion
```

Logistic Regression was tuned with **30 trials** using:

```text
C
```

on a logarithmic scale between `0.0001` and `100`.

Random Forest is much more computationally expensive, so it was tuned using **15 trials**.

Its search space included:

```text
n_estimators
max_depth
min_samples_leaf
max_features
```

For the Random Forest experiment:

```yaml
cv:
  n_jobs: -1
```

was used to allow cross-validation jobs to use the available CPU cores.

---

### Hyperparameter tuning results

| Model | Default CV mean ± std | Tuned nested CV mean ± std | Best trial score | Optimism | Chosen hyperparameters |
|---|---:|---:|---:|---:|---|
| Decision Tree | 0.610 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | `max_depth=6`, `min_samples_leaf=101`, `criterion=gini` |
| Logistic Regression | 0.672 ± 0.013 | 0.673 ± 0.016 | 0.678 | +0.002 | `C=0.0715` |
| **Random Forest** | 0.647 ± 0.016 | **0.681 ± 0.014** | 0.683 | -0.001 | `n_estimators=137`, `max_depth=None`, `min_samples_leaf=33`, `max_features=log2` |

---

### Decision Tree tuning

The default Decision Tree produced:

```text
Train mean      = 0.695
Validation mean = 0.610
Gap mean        = +0.085
```

After tuning, nested CV produced:

```text
Train mean      = 0.685
Validation mean = 0.672
Std             = 0.017
Gap mean        = +0.013
```

The chosen hyperparameters were:

```text
max_depth = 6
min_samples_leaf = 101
criterion = gini
```

The validation accuracy increased by approximately:

```text
0.672 - 0.610 = +0.062
```

The train-validation gap also decreased substantially.

This shows that the poor performance of the default tree was strongly related to its unrestricted complexity. Hyperparameter tuning reduced overfitting and produced much better generalisation.

---

### Logistic Regression tuning

The default Logistic Regression produced:

```text
Validation mean = 0.672
Std             = 0.013
Gap mean        = +0.003
```

After tuning:

```text
Validation mean = 0.673
Std             = 0.016
Gap mean        = +0.004
```

The selected regularisation value was approximately:

```text
C = 0.0715
```

The improvement was only:

```text
0.673 - 0.672 = +0.001
```

This difference is much smaller than the fold-to-fold variability.

Therefore, tuning did not meaningfully improve Logistic Regression.

This is consistent with the fact that the default Logistic Regression was already stable and showed very little evidence of overfitting.

---

### Random Forest tuning

The default Random Forest produced:

```text
Train mean      = 0.732
Validation mean = 0.647
Std             = 0.016
Gap mean        = +0.085
```

After tuning, nested CV produced:

```text
Train mean      = 0.690
Validation mean = 0.681
Std             = 0.014
Gap mean        = +0.009
```

The final search on the complete development set selected:

```text
n_estimators = 137
max_depth = None
min_samples_leaf = 33
max_features = log2
```

`max_depth=None` means that the depth itself was not explicitly limited. However, the much larger `min_samples_leaf=33` strongly restricts the complexity of individual trees by preventing leaves based on very small groups of observations.

The Random Forest improved by approximately:

```text
0.681 - 0.647 = +0.034
```

and its train-validation gap decreased from approximately:

```text
0.085 → 0.009
```

This indicates substantially better generalisation after tuning.

---

### Which model benefited most from tuning?

The improvements were approximately:

| Model | Improvement after tuning |
|---|---:|
| Decision Tree | +0.062 |
| Logistic Regression | +0.001 |
| Random Forest | +0.034 |

The Decision Tree gained the most from tuning.

The Random Forest also improved meaningfully.

Logistic Regression gained almost nothing because its default configuration was already relatively stable.

For Decision Tree and Random Forest, the improvement is larger than their fold-to-fold standard deviation.

For Logistic Regression, the improvement is much smaller than its standard deviation and should not be interpreted as a meaningful gain.

---

### Did tuning change the best model?

Yes.

Before tuning, Logistic Regression had the highest mean CV accuracy:

```text
Logistic Regression = 0.672 ± 0.013
```

After tuning, Random Forest obtained the highest nested-CV mean:

```text
Random Forest = 0.681 ± 0.014
Logistic Regression = 0.673 ± 0.016
Decision Tree = 0.672 ± 0.017
```

Therefore, the tuned Random Forest has the **highest observed mean accuracy**.

However, the difference between Random Forest and the other tuned models is relatively small compared with the fold-to-fold variability.

For that reason, the result does not prove that Random Forest is decisively superior. It is simply the model with the highest observed honest nested-CV estimate in the current experiment.

---

### Random Forest classification results

Using the nested-CV out-of-fold predictions, the tuned Random Forest produced:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| 0 | 0.69 | 0.78 | 0.73 | 3170 |
| 1 | 0.67 | 0.57 | 0.61 | 2601 |

Overall out-of-fold accuracy was approximately:

```text
0.68
```

The class 1 recall of `0.57` is higher than the tuned Logistic Regression class 1 recall of approximately `0.50`.

---

### Fairness audit for tuned Random Forest

The fairness report is calculated from nested-CV out-of-fold predictions on the development set.

For the tuned Random Forest:

| Race | Model FPR |
|---|---:|
| African-American | 0.30 |
| Asian | 0.14 |
| Caucasian | 0.16 |
| Hispanic | 0.17 |
| Native American | 0.17 |
| Other | 0.18 |
| unknown | 0.20 |

For comparison, COMPAS's own score produced:

| Race | COMPAS FPR |
|---|---:|
| African-American | 0.45 |
| Asian | 0.10 |
| Caucasian | 0.23 |
| Hispanic | 0.23 |
| Native American | 0.17 |
| Other | 0.14 |
| unknown | 0.33 |

The FPR varies across groups.

Some groups have very small numbers of negative observations, particularly Asian and Native American observations, so these individual rates should not be treated as stable population estimates.

The fairness report is therefore an audit of model behaviour, not a complete fairness assessment.

---

### Final tuned model

The current best observed nested-CV result was obtained by the tuned Random Forest.

The final model was refitted on:

```text
5771 development rows
```

with:

```text
n_estimators = 137
max_depth = None
min_samples_leaf = 33
max_features = log2
```

The final locked test set contains:

```text
1443 rows
```

and remains **set aside and unevaluated**.

---

## Environment setup

You only need to create the virtual environment and install the dependencies once per machine.

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Windows -- PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
.\venv\Scripts\Activate.ps1
```

### Windows -- cmd.exe

```cmd
python -m venv venv
venv\Scripts\activate.bat
python -m pip install -r requirements.txt
```

Once the environment is active, the prompt should begin with:

```text
(venv)
```

To leave the environment later:

```text
deactivate
```

---

### Every time after the first

Creating the environment and installing the packages only needs to happen once.

On macOS/Linux:

```bash
source venv/bin/activate
python main.py
```

On Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

When `requirements.txt` changes, install the new dependencies again.

For example, Week 5 added Optuna:

```powershell
python -m pip install -r requirements.txt
```

The installed Optuna version can be checked with:

```powershell
python -c "import optuna; print(optuna.__version__)"
```

---

## Environment Troubleshooting

### PowerShell blocks the venv activation script

If PowerShell repeatedly blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

On restricted school or lab machines, Git Bash or `cmd.exe` can also be used.

---

### File access problems on Windows

If Windows blocks Python or the terminal from accessing files in Desktop, Documents or Pictures, check:

```text
Windows Security
→ Virus & threat protection
→ Manage ransomware protection
```

and:

```text
Settings
→ Privacy & security
→ File system
```

---

## Running the pipeline

With the virtual environment active, from the project's root folder:

```bash
python main.py
```

The current pipeline performs:

```text
load config
    ↓
load raw data
    ↓
row-preserving cleaning
    ↓
training-only duplicate removal
    ↓
features / target / audit columns
    ↓
lock final test set
    ↓
build preprocessing + model pipeline
    ↓
default stratified 5-fold CV
    ↓
if tuning enabled:
    nested cross-validation
        ↓
    Optuna tuning on full development set
        ↓
    tuned out-of-fold predictions
    ↓
classification report
    ↓
fairness report
    ↓
refit final model on all development rows
    ↓
save results
```

The terminal prints:

- default cross-validation results;
- train, validation and gap for each fold;
- nested CV results when tuning is enabled;
- hyperparameters chosen in each outer fold;
- best final hyperparameters;
- Optuna's top trials;
- optimism check;
- out-of-fold classification report;
- false-positive rate by race;
- COMPAS comparison;
- confirmation of the final refit;
- locked-test row count;
- saved results path.

All output is also saved to a timestamped file in:

```text
results/
```

For example:

```text
results/run_YYYYMMDD_HHMMSS.txt
```

The `results/` directory contains generated outputs and is not intended to be tracked as source code.

---

## Current tuning configuration

The current model configuration is:

```yaml
model:
  type: "random_forest"
  params:
    random_state: 42
```

The current tuning configuration includes:

```yaml
tuning:
  enabled: true
  n_trials: 15
  n_splits: 5
  random_state: 42

  search_spaces:
    decision_tree:
      model__max_depth: {type: int, low: 1, high: 20}
      model__min_samples_leaf: {type: int, low: 1, high: 200, log: true}
      model__criterion: {type: categorical, choices: ["gini", "entropy"]}

    logistic_regression:
      model__C: {type: float, low: 0.0001, high: 100, log: true}

    random_forest:
      model__n_estimators: {type: int, low: 50, high: 200}
      model__max_depth: {type: categorical, choices: [null, 5, 10, 15, 20]}
      model__min_samples_leaf: {type: int, low: 1, high: 50, log: true}
      model__max_features: {type: categorical, choices: ["sqrt", "log2", 0.5, 1.0]}
```

The Random Forest uses fewer trials because each trial is considerably more computationally expensive than Logistic Regression or a single Decision Tree.

---

## Push to GitHub via Terminal

Standard workflow from the project's root folder:

```bash
git status
git add .
git commit -m "short description of what changed"
git push
```

Always use:

```bash
git status
```

before committing to check exactly which files will be included.

If `git push` asks for a password and rejects the normal GitHub account password, GitHub requires authentication through a Personal Access Token or another configured credential method for HTTPS Git operations.

---

## Dataset

See:

```text
data/README.md
```

for the problem description and complete data dictionary.