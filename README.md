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
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |

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

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
