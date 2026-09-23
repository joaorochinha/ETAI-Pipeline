# Baseline Predictive Pipeline -- ETAI
**Name:** João Rocha  
**Student Number:** 20260600


# Logistic Regression Analysis

The Logistic Regression model achieved a training accuracy of approximately **67.9%** and a test accuracy of approximately **67.7%**.

The difference between training and test accuracy is very small, with a gap of approximately **0.1 percentage points**. This suggests that the model does not show clear evidence of overfitting and generalises well to unseen data.

For class 0, the model achieved a precision of **0.69**, recall of **0.74**, and an F1-score of **0.71**.

For class 1, the model achieved a precision of **0.66**, recall of **0.60**, and an F1-score of **0.63**.

The model performs slightly better for class 0 than for class 1. In particular, the lower recall for class 1 indicates that the model has more difficulty identifying positive cases.

Compared with the Decision Tree, Logistic Regression generalises better to unseen data and achieves a higher test accuracy. The Decision Tree has a much larger gap between training and test accuracy, which indicates overfitting.

# Decision Tree Analysis

The Decision Tree achieved a training accuracy of approximately **82.9%**, but its test accuracy decreased to approximately **62.6%**.

The difference between training and test accuracy is approximately **20.3 percentage points**. This large gap provides clear evidence of overfitting. The model performs considerably better on the data used during training than on unseen data.

The model also performs differently across the two classes.

For class 0, the Decision Tree achieved a precision of **0.63**, recall of **0.74**, and an F1-score of **0.69**.

For class 1, precision was **0.61**, recall was only **0.48**, and the F1-score was **0.54**.

The recall of 0.48 for class 1 means that the model correctly identifies only around 48% of the actual positive observations. Therefore, the model has considerable difficulty identifying class 1 cases.

The results suggest that the Decision Tree is too complex in its current configuration and is fitting characteristics that are specific to the training dataset rather than patterns that generalise well to unseen data.

# Logistic Regression vs Decision Tree

The two models show substantially different behaviour.

The Decision Tree obtains a much higher training accuracy than the Logistic Regression (**82.9% vs 67.9%**). However, this improvement does not generalise to unseen data.

On the test set, Logistic Regression achieved an accuracy of **67.7%**, while the Decision Tree achieved only **62.6%**.

The train-test gap is particularly important. Logistic Regression has almost no difference between its training and test performance, while the Decision Tree has a gap of approximately **20.3 percentage points**. This indicates that the Decision Tree is substantially overfitting the training data.

The Logistic Regression also performs better for class 1. Its recall for class 1 is **0.60**, compared with only **0.48** for the Decision Tree. Its class 1 F1-score is also higher (**0.63 vs 0.54**).

Overall, these results show that a higher training accuracy does not necessarily mean that a model is better. The Decision Tree fits the training data more closely, but its performance decreases considerably on unseen observations. In contrast, Logistic Regression has lower training accuracy but more stable performance between the training and test sets.

Logistic Regression is better because it does not show overfitting and performs better on the test data.






This is the **starting point** for your semester project: a small but *complete* predictive pipeline -- every piece a real project needs (entry point, config, data loading, preprocessing, model, evaluation), just kept as simple as possible for now.

The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.

It has some **deliberately weak spots**. Part of your work this
semester is finding them and making them better -- see the pipeline progress table below, which tracks what changes and why as the weeks
go on.

## Project structure

```
.
├── main.py                  # entry point: run the whole pipeline
├── config.yaml               # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py               # loading
│   ├── data_diagnostics.py   # missingness-mechanism test, domain-rule checks, duplicate check (new week 3)
│   ├── preprocessing.py      # leak-safe cleaning, deployable preprocessing pipeline, and train/test split (week 3 grew this file's job well beyond just the split -- same file, same name as week 2)
│   ├── model.py               # model construction
│   ├── evaluate.py           # accuracy  + fairness check
│   └── results.py            # saves each run's report to disk
├── results/                  # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md              # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (count encoding + robust scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. |

## Preprocessing decisions

*(New this week -- written straight from the diagnosis in `Practical/W3/notebooks/01_eda_introduction.ipynb` and the empirical grid in `02_preprocessing.ipynb`. Full walkthrough lives in those two notebooks; this is the summary.)*

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen empirically -- 4 encoders (one-hot, ordinal, count, target) × 4 scalers (none, standard, min-max, robust), scored by mean accuracy across 15 repeated train/test splits with logistic regression. **Target encoding + standard scaling won**, though a paired comparison against the runner-up (same 15 splits, per-split difference) showed the margin was within noise -- see `02_preprocessing.ipynb`'s grid + paired-comparison cells for the full table and the check itself.

## Environment setup

You only need to do this once per machine.

### macOS / Linux
```bash
python3 -m venv venv                 # creates an isolated Python environment in a folder called "venv"
source venv/bin/activate             # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Environment Troubleshooting

Two Windows issues come up often enough to note here -- if you hit either, this saves you re-diagnosing it from scratch.

**PowerShell blocks the venv activation script, every new terminal.** The `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` line above only fixes it for that one terminal window -- close it and it's back. For a fix that actually sticks across sessions, run this **once**, instead:
```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```
If it still doesn't stick (common on locked-down school/lab machines with a Group Policy that resets it on every logon), skip PowerShell entirely: use **Git Bash** (`source venv/Scripts/activate`) or **cmd.exe** (`venv\Scripts\activate.bat`) instead -- neither is affected by PowerShell's execution policy.

**Windows blocks the terminal/Python from reading or writing files in Documents (or Desktop/Pictures).** Shows up as an "Access is denied" error, or a silent failure to create/update a file, only when the project sits inside one of those folders. Two independent settings can cause this -- check both:
- **Windows Security -> Virus & threat protection -> Manage ransomware protection** -- turn off **Controlled folder access**, or add your terminal/Python/editor to its allowed-apps list.
- **Settings -> Privacy & security -> File system** -- make sure the terminal/Python has access.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, diagnoses and cleans the data (week 3's `data_diagnostics.py`/`preprocessing.py`), preprocesses and trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises.
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the pipeline, and isn't tracked in git (see `.gitignore`) since it's generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Push to GitHub via Terminal

Standard workflow, from the project's root folder, with the venv active:
```bash
git add .
git commit -m "short description of what changed"
git push
```

**If `git push` asks for a password and rejects your normal GitHub password:** GitHub no longer accepts account passwords for git over HTTPS -- you need a **Personal Access Token (PAT)** instead.
1. On GitHub: **Settings -> Developer settings -> Personal access tokens -> Tokens (classic)** -> **Generate new token**, with at least `repo` scope.
2. When `git push` prompts for a password, paste the token instead (username stays your GitHub username).
3. So you're not asked every time: `git config --global credential.helper manager` (Windows, usually already set up by Git for Windows) or `git config --global credential.helper store` (caches it in plaintext -- fine on a personal machine, not a shared one).

Alternative: set up an SSH key once (`ssh-keygen -t ed25519`, then add the public key under **GitHub -> Settings -> SSH and GPG keys**) and use the repo's SSH remote URL (`git@github.com:...`) instead of HTTPS -- no token to manage or renew.

## Dataset

See `data/README.md`.
