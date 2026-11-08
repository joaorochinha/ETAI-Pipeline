"""Saving each run: the full report (one text file per run) and one row in the scoreboard CSV."""
import os

import pandas as pd
import yaml


def save_run(results_dir: str, run: str, config: dict, report_text: str) -> str:
    """
    Writes the run's settings, full report and a copy of the whole config used to
    `results_dir/run_<run>.txt` (the folder is created if needed) and returns the path.
    `run` is the run id (a timestamp) made in `main.py`, the same one as in the scoreboard.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"run_{run}.txt")

    cv = config["cv"]
    tuning = config.get("tuning", {})
    tuning_line = (f"Optuna, {tuning['n_trials']} trials x {tuning['n_splits']} folds, nested CV  "
                   f"random_state={tuning['random_state']}" if tuning.get("enabled") else "off")
    selection = config.get("feature_selection", {})
    selection_line = (f"VoteSelector  min_votes={selection.get('min_votes', 2)}"
                      + ("  (tuned: select__min_votes in the search space)" if tuning.get("enabled") else "")
                      if selection.get("enabled") else "off")
    header = (
        f"Run: {run}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params')}\n"
        f"Preprocessing: encoder={config['preprocessing']['encoder']}  "
        f"scaler={config['preprocessing']['scaler']}  "
        f"numeric imputation={config['preprocessing'].get('imputation', {}).get('numeric_strategy', 'median')}\n"
        f"CV: {cv['n_splits']} stratified folds  shuffle={cv.get('shuffle', True)}  "
        f"random_state={cv.get('random_state')}\n"
        f"Metrics: {', '.join(cv['metrics'])}   selection metric: {cv['selection_metric']}\n"
        f"Tuning: {tuning_line}\n"
        f"Feature selection: {selection_line}\n"
        f"Locked test set: size={config['test_set']['size']}  "
        f"random_state={config['test_set']['random_state']}  (not evaluated)\n"
        + "=" * 60 + "\n\n"
    )

    with open(path, "w") as f:
        f.write(header + report_text)
        f.write("\n" + "=" * 60 + "\nConfig used for this run:\n\n")
        f.write(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    return path


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Appends one run to the CSV at `path` (created if needed) and returns all runs."""
    runs = pd.read_csv(path, dtype={"run": str}) if os.path.exists(path) else pd.DataFrame()
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    runs.to_csv(path, index=False)
    return runs