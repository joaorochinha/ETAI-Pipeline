# Baseline Predictive Pipeline -- ETAI

**Name:** João Rocha  
**Student Number:** 20260600

---

# Week 2

## Logistic Regression Analysis

The Logistic Regression model achieved a training accuracy of approximately **67.9%** and a test accuracy of approximately **67.7%**.

The difference between training and test accuracy was very small, with a gap of approximately **0.1 percentage points**. This suggested that the model did not show clear evidence of overfitting and generalised well to unseen data.

For class 0, the model achieved:

- Precision: **0.69**
- Recall: **0.74**
- F1-score: **0.71**

For class 1, the model achieved:

- Precision: **0.66**
- Recall: **0.60**
- F1-score: **0.63**

The model performed slightly better for class 0 than for class 1. In particular, the lower recall for class 1 showed that the model had more difficulty identifying positive cases.

Compared with the Decision Tree, Logistic Regression generalised better to unseen data and achieved a higher test accuracy.

---

## Decision Tree Analysis

The Decision Tree achieved a training accuracy of approximately **82.9%**, while its test accuracy was approximately **62.6%**.

The difference between training and test accuracy was approximately **20.3 percentage points**.

This large gap provided clear evidence of overfitting. The model performed considerably better on the data used during training than on unseen data.

For class 0, the model achieved:

- Precision: **0.63**
- Recall: **0.74**
- F1-score: **0.69**

For class 1, the model achieved:

- Precision: **0.61**
- Recall: **0.48**
- F1-score: **0.54**

The recall of 0.48 for class 1 means that the model correctly identified only around 48% of the actual positive observations.

The results suggested that the Decision Tree was too complex in its current configuration and was fitting characteristics specific to the training dataset rather than patterns that generalised well to unseen data.

---

## Logistic Regression vs Decision Tree

The two models showed substantially different behaviour.

The Decision Tree obtained a much higher training accuracy than Logistic Regression:

- Decision Tree: **82.9%**
- Logistic Regression: **67.9%**

However, this improvement did not generalise to unseen data.

On the test set:

- Logistic Regression: **67.7%**
- Decision Tree: **62.6%**

The train-test gap was particularly important.

- Logistic Regression gap: approximately **0.1 percentage points**
- Decision Tree gap: approximately **20.3 percentage points**

This indicated that the Decision Tree substantially overfit the training data.

Logistic Regression also performed better for class 1:

- Logistic Regression recall: **0.60**
- Decision Tree recall: **0.48**

Its class 1 F1-score was also higher:

- Logistic Regression: **0.63**
- Decision Tree: **0.54**

Overall, these results showed that higher training accuracy does not necessarily mean that a model is better.

The Decision Tree fitted the training data more closely, but its performance decreased considerably on unseen observations.

Logistic Regression had lower training accuracy but much more stable performance between the training and test sets.

---

# Week 3 - Data Cleaning and EDA

During Week 3, the dataset was analysed in more detail before model training.

The objective was to diagnose data-quality problems before deciding how they should be handled in the predictive pipeline.

The analysis investigated:

- placeholder values;
- invalid numerical values;
- missing-value mechanisms;
- inconsistent categorical values;
- duplicate observations;
- redundant variables;
- multicollinearity.

---

## Invalid and Missing Values

Some observations contained values that were technically present in the dataset but were not valid values.

Examples included:

- invalid ages;
- invalid COMPAS decile scores;
- negative juvenile counts;
- unrealistic prior-offence counts;
- placeholder strings such as `-`, `?`, `NA`, and `n/a`.

These values were converted to proper missing values (`NaN`) before imputation.

The main validity rules used were:

| Variable | Rule |
|---|---|
| `age` | 18 to 100 |
| `decile_score` | 1 to 10 |
| `juv_fel_count` | >= 0 |
| `priors_count` | 0 to 60 |

---

## Missingness Mechanisms

The Week 3 exploratory analysis investigated whether missingness appeared to be MCAR or related to other variables.

The resulting preprocessing decisions were stored in `config.yaml`.

The current configuration uses:

| Variable | Strategy |
|---|---|
| `age` | median imputation |
| `juv_fel_count` | median imputation |
| `priors_count` | median imputation + missingness indicator |
| `c_charge_degree` | most-frequent imputation + missingness indicator |
| `sex` | most-frequent imputation |

The missingness indicators used by the pipeline are:

```text
priors_count_was_missing
c_charge_degree_was_missing
```

These indicators are created before imputation so that the model can still use information about whether the original value was missing.

---

## Age and Age Category Investigation

The relationship between `age` and `age_cat` was analysed as an additional data-quality check.

The investigation found a small number of inconsistent observations, including invalid age values and cases where a valid numerical age did not correspond to the expected age category.

This was useful for understanding the quality of the dataset.

However, the final Week 4 `clean_dataset()` implementation follows the generic configuration-driven cleaning recipe and does not include a hardcoded rule that reconstructs `age_cat` from `age`.

---

## COMPAS Score Consistency Investigation

The relationship between `decile_score` and `score_text` was also investigated.

Different representations such as:

```text
LOW
low
Low
```

were identified and categorical spellings were standardised.

The analysis also investigated whether `score_text` was consistent with `decile_score`.

This was treated as an exploratory data-quality investigation.

The current generic cleaning pipeline canonicalises category spellings using the mappings stored in `config.yaml`.

---

## Duplicate Observations

Before duplicate removal, the dataset contained:

- **72 exact duplicate rows**
- **72 repeated IDs**

From Week 4 onward, duplicate removal is deliberately separated from general cleaning.

`clean_dataset()` is row-preserving:

```text
same rows in
same rows out
same order
```

This makes the function safe to apply to future prediction data, where every input row requires a prediction.

Duplicate removal is handled separately by:

```python
drop_duplicate_rows()
```

This function is applied only to the labelled dataset before the development/test split.

After duplicate removal:

- Rows: **7286 -> 7214**
- Duplicate observations removed: **72**

---

## Missing Values After Cleaning

After invalid values and placeholders were converted to `NaN`, the main missing-value percentages were approximately:

| Variable | Missing % |
|---|---:|
| `priors_count` | 6.97% |
| `c_charge_degree` | 3.17% |
| `juv_fel_count` | 3.06% |
| `age` | 2.09% |
| `race` | 2.00% |
| `sex` | 1.50% |
| `decile_score` | 0.08% |

These missing values are not simply deleted.

Instead, imputation is performed inside the sklearn preprocessing pipeline so that fitted values are learned only from the appropriate training data.

---

## Redundant Variables

Three variables were identified as redundant during the Week 3 analysis:

```text
prior_offenses
age_in_months
juvenile_total
```

These variables are removed before modelling.

The purpose is to avoid keeping variables that duplicate information already represented elsewhere in the dataset.

---

# Week 2 vs Week 3 Model Comparison

The same two models were evaluated again after the Week 3 data-quality and preprocessing changes.

| Model | Week 2 Train | Week 2 Test | Week 2 Gap | Week 3 Train | Week 3 Test | Week 3 Gap |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.679 | 0.677 | 0.001 | 0.675 | 0.657 | 0.018 |
| Decision Tree | 0.829 | 0.626 | 0.203 | 0.792 | 0.612 | 0.180 |

The Week 3 pipeline produced slightly lower test accuracy for both models.

However, the Decision Tree showed a smaller train-test gap than in Week 2.

Its gap decreased from:

```text
0.203 -> 0.180
```

This suggested that its overfitting was reduced slightly, although it remained substantial.

Logistic Regression continued to generalise better than the Decision Tree.

The objective of the additional cleaning was primarily to improve data consistency and preprocessing quality rather than to guarantee an increase in predictive accuracy.

---

# Week 4 - Leak-Safe Preprocessing Recipe

Week 4 reorganised the preprocessing pipeline so that the distinction between:

1. stateless cleaning;
2. training-data-only decisions;
3. fitted preprocessing operations;

is explicit.

The main rule is:

> Held-out observations must behave like future unseen observations.

Therefore, any operation that learns information from data must only be fitted using the appropriate training observations.

---

## Row-Preserving Cleaning

`clean_dataset()` now performs only stateless cleaning operations.

These include:

- conversion of placeholders to `NaN`;
- conversion of numeric text columns;
- invalid-value detection;
- categorical canonicalisation;
- removal of redundant columns.

Importantly:

```text
clean_dataset():
rows in = rows out
```

Duplicate observations are no longer removed inside this function.

---

## Training-Only Duplicate Removal

Duplicate removal was moved into:

```python
drop_duplicate_rows()
```

This operation happens before the development/test split.

The objective is to prevent the same observation or person from appearing in both the development data and the locked test set.

It is not applied to future prediction data because every incoming row must receive a prediction.

---

## Development and Locked Test Set

The previous:

```python
split_train_test()
```

was replaced with:

```python
split_dev_test()
```

The dataset is divided into:

- **80% development set**
- **20% locked test set**

using:

```yaml
split:
  test_size: 0.2
  random_state: 42
```

The development set is the data available for:

- training;
- model comparison;
- cross-validation;
- hyperparameter tuning.

The locked test set should not be used to make modelling or preprocessing decisions.

It represents the final unseen evaluation data.

---

## Missingness Indicators

Before imputation, the pipeline creates missingness flags for variables diagnosed as requiring them.

The current configuration is:

```yaml
mnar_indicator_sources:
  - priors_count
  - c_charge_degree
```

This produces:

```text
priors_count_was_missing
c_charge_degree_was_missing
```

The flags remain available after the original missing values are imputed.

---

## Imputation

Imputation is performed inside the sklearn `Pipeline`.

The current strategies are:

```yaml
imputation:
  numeric_strategy: "median"
  categorical_strategy: "most_frequent"
```

This is leak-safe because the imputation values are fitted only on the training data available to that fitting operation.

---

## Categorical Encoding

The current categorical encoder is:

```yaml
encoder: "target"
```

The pipeline uses scikit-learn's `TargetEncoder`.

The encoder uses internal stratified cross-fitting:

```python
StratifiedKFold(
    5,
    shuffle=True,
    random_state=42
)
```

The categorical features are:

```text
sex
age_cat
c_charge_degree
```

Target encoding produces a compact numeric representation of categorical variables.

Alternative encoders remain available in the code:

```text
onehot
ordinal
count
target
```

---

## Scaling

The current scaler is:

```yaml
scaler: "robust"
```

`RobustScaler` uses the median and interquartile range.

This is useful for numerical count variables with extreme observations because extreme values have less influence on the scale than they would with mean/standard-deviation scaling.

Alternative scalers remain available:

```text
none
standard
minmax
robust
```

The current preprocessing recipe is therefore:

```text
Target Encoding
+
Robust Scaling
```

---

## Model Construction

The Week 4 `model.py` supports four classifiers:

```text
dummy
logistic_regression
decision_tree
random_forest
```

The model is selected through `config.yaml`.

For example:

```yaml
model:
  type: "logistic_regression"
  params:
    max_iter: 2000
```

This allows different models to be tested without changing the Python source code.

---

# Week 4 Model Results

After implementing the Week 4 preprocessing pipeline, Logistic Regression and Decision Tree were evaluated again.

The current evaluator still prints the development-set score as `Train accuracy`, because the model is fitted on the complete development set before evaluation.

| Model | Development Accuracy | Locked Test Accuracy | Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.676 | **0.658** | **0.018** |
| Decision Tree | 0.687 | 0.602 | 0.085 |

---

## Logistic Regression

Current results:

```text
Development accuracy: 0.676
Test accuracy:        0.658
Gap:                  0.018
```

Classification report on the locked test set:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0 | 0.65 | 0.80 | 0.72 |
| 1 | 0.66 | 0.48 | 0.56 |

The development-test gap is small.

This indicates relatively stable performance between the data used to fit the pipeline and the locked test observations.

The model still has more difficulty identifying class 1 than class 0, particularly in recall.

---

## Decision Tree

Current results:

```text
Development accuracy: 0.687
Test accuracy:        0.602
Gap:                  0.085
```

Classification report on the locked test set:

| Class | Precision | Recall | F1-score |
|---|---:|---:|---:|
| 0 | 0.62 | 0.72 | 0.66 |
| 1 | 0.57 | 0.46 | 0.51 |

The Decision Tree has a higher development accuracy than Logistic Regression, but lower performance on the locked test set.

It also has a larger development-test gap.

This indicates that the Decision Tree continues to show more evidence of overfitting than Logistic Regression.

---

## Current Model Comparison

| Model | Development | Test | Gap |
|---|---:|---:|---:|
| Logistic Regression | 0.676 | **0.658** | **0.018** |
| Decision Tree | **0.687** | 0.602 | 0.085 |

Based on these results, Logistic Regression remains the strongest of the two currently evaluated models because:

- it achieves higher locked-test accuracy;
- it has a much smaller development-test gap;
- its performance is more stable on unseen observations.

However, the locked test set should not be repeatedly used for model selection.

The next stage of the project introduces cross-validation so that model and preprocessing decisions can be evaluated using the development set without repeatedly consulting the locked test set.

---

# Fairness Check

The pipeline also reports the false-positive rate (FPR) by race.

For the current Logistic Regression model:

| Race | FPR |
|---|---:|
| African-American | 0.28 |
| Asian | 0.00 |
| Caucasian | 0.14 |
| Hispanic | 0.11 |
| Native American | 0.00 |
| Other | 0.19 |

For comparison, COMPAS's own score produced:

| Race | FPR |
|---|---:|
| African-American | 0.44 |
| Asian | 0.00 |
| Caucasian | 0.24 |
| Hispanic | 0.16 |
| Native American | 1.00 |
| Other | 0.20 |

The results show that false-positive rates differ across racial groups.

However, some groups have extremely small sample sizes in the relevant subset, particularly Asian and Native American observations, so those individual rates should not be interpreted as stable estimates.

The fairness analysis is therefore an audit of model behaviour rather than a complete fairness assessment.

---

# Project Description

This project is a small but complete predictive machine-learning pipeline.

The task is to predict two-year recidivism using ProPublica's COMPAS dataset.

The project progressively develops a realistic predictive workflow containing:

- configuration;
- data loading;
- cleaning;
- preprocessing;
- model construction;
- evaluation;
- fairness auditing;
- result persistence.

The pipeline intentionally begins simple and becomes more robust throughout the semester.

See:

```text
data/README.md
```

for the full dataset description and data dictionary.

---

# Project Structure

```text
.
├── main.py                  # pipeline entry point
├── config.yaml              # configuration and modelling choices
├── requirements.txt
├── src/
│   ├── data.py              # data loading
│   ├── preprocessing.py     # cleaning, duplicates, feature split,
│   │                        # preprocessing and dev/test split
│   ├── model.py             # model construction
│   ├── evaluate.py          # performance and fairness evaluation
│   └── results.py           # saves run reports
├── results/                 # generated run reports
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md
```

`src/data_diagnostics.py` was used during the Week 3 diagnostic stage.

From Week 4 onward, the reusable cleaning rule `flag_invalid_values()` lives inside `src/preprocessing.py`, while the exploratory diagnostic functions are no longer required by the production pipeline.

---

# Pipeline Progress

| Week | Practical focus | Main pipeline changes |
|---|---|---|
| 2 | Introduction and baseline pipeline | Initial runnable pipeline, simple train/test split, Logistic Regression baseline, Decision Tree comparison, classification report, fairness FPR report and saved results |
| 3 | EDA and data diagnosis | Missingness investigation, invalid-value rules, categorical canonicalisation, duplicate diagnosis, multicollinearity analysis, mechanism-matched imputation and missingness indicators |
| 4 | Leak-safe preprocessing | `clean_dataset()` becomes row-preserving; `drop_duplicate_rows()` becomes a separate training-only operation; `split_dev_test()` creates development and locked test sets; sklearn `TargetEncoder` with stratified cross-fitting; Robust Scaling; `data_diagnostics.py` removed from the runtime pipeline; Dummy and Random Forest model options added |

---

# Current Configuration

The current principal preprocessing configuration is:

```yaml
preprocessing:
  encoder: "target"
  scaler: "robust"
  random_state: 42

  numeric_features:
    - age
    - juv_fel_count
    - juv_misd_count
    - juv_other_count
    - priors_count

  categorical_features:
    - sex
    - age_cat
    - c_charge_degree

  mnar_indicator_sources:
    - priors_count
    - c_charge_degree

  imputation:
    numeric_strategy: "median"
    categorical_strategy: "most_frequent"
```

The current baseline model is:

```yaml
model:
  type: "logistic_regression"

  params:
    max_iter: 2000
```

---

# Environment Setup

You only need to create the virtual environment and install the dependencies once per machine.

## macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Windows -- PowerShell

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate the environment again:

```powershell
venv\Scripts\activate
```

## Windows -- cmd.exe

```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

When the environment is active, the terminal prompt should begin with:

```text
(venv)
```

---

# Every Time You Return to the Project

You do not need to recreate the environment.

On Windows:

```powershell
venv\Scripts\activate
python main.py
```

On macOS/Linux:

```bash
source venv/bin/activate
python main.py
```

---

# Environment Troubleshooting

## PowerShell blocks activation

If PowerShell repeatedly blocks the activation script, you can run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

On restricted machines, Git Bash or `cmd.exe` can also be used.

---

## File Access Problems on Windows

If Windows blocks Python or the terminal from accessing files in Desktop, Documents or Pictures, check:

```text
Windows Security
-> Virus & threat protection
-> Manage ransomware protection
```

and:

```text
Settings
-> Privacy & security
-> File system
```

---

# Running the Pipeline

With the virtual environment active, run:

```bash
python main.py
```

The pipeline performs the following steps:

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
development / locked test split
    ↓
preprocessing pipeline
    ↓
model
    ↓
predictions
    ↓
performance evaluation
    ↓
fairness report
    ↓
save results
```

The terminal prints:

- development/train accuracy;
- locked test accuracy;
- development-test gap;
- classification report;
- false-positive rate by race;
- COMPAS comparison.

The complete output is also saved automatically to:

```text
results/
```

For example:

```text
results/run_20261001_193930.txt
```

The `results/` directory contains generated outputs and is not intended to be part of the source code.

---

# Changing Models

The model can be changed directly in `config.yaml`.

## Logistic Regression

```yaml
model:
  type: "logistic_regression"
  params:
    max_iter: 2000
```

## Decision Tree

```yaml
model:
  type: "decision_tree"
  params: {}
```

## Random Forest

```yaml
model:
  type: "random_forest"
  params: {}
```

## Dummy Classifier

```yaml
model:
  type: "dummy"
  params: {}
```

Model-specific parameters must only be supplied to models that support them.

For example:

```text
max_iter
```

is a Logistic Regression parameter and should not be passed to the Decision Tree.

---

# Git Workflow

From the project root:

```bash
git status
git add .
git commit -m "short description of what changed"
git push
```

Before committing, `git status` can be used to confirm exactly which files were changed, added or deleted.

---

# Updating the Repository

When the professor updates the upstream repository, first update the fork and then pull the changes locally.

Typical workflow:

```bash
git pull
```

Always check the current branch before working:

```bash
git branch
```

The active branch is marked with:

```text
*
```

---

# Dataset

See:

```text
data/README.md
```

for the problem description and complete data dictionary.

---

# Next Step

The Week 4 pipeline now provides a leak-safe preprocessing recipe and a locked test set.

The next stage is:

```text
03_cross_validation.ipynb
```

Cross-validation will allow model and preprocessing decisions to be evaluated using only the development set, without repeatedly using the locked test set for model selection.