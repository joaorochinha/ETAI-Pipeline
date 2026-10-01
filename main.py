"""
Entry point for the predictive pipeline.

Run with:
    python main.py

Week 4 pipeline:
    load config
    -> load data
    -> clean data
    -> remove duplicate training rows
    -> split features/target
    -> create development + locked test sets
    -> preprocess + train model
    -> evaluate
    -> save results
"""

import yaml

from sklearn.pipeline import Pipeline

from src.data import load_data

from src.preprocessing import (
    clean_dataset,
    drop_duplicate_rows,
    split_features_target,
    build_preprocessor,
    split_dev_test,
)

from src.model import build_model
from src.evaluate import evaluate, fairness_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():

    config = load_config()

    # ---------------------------------------------------------
    # 1. Load raw data
    # ---------------------------------------------------------

    df_raw = load_data(
        config["data"]["path"]
    )

    # ---------------------------------------------------------
    # 2. Stateless cleaning
    # ---------------------------------------------------------
    # Cleaning preserves the number and order of rows.

    df_clean = clean_dataset(
        df_raw,
        config["diagnostics"]
    )

    # ---------------------------------------------------------
    # 3. Remove duplicate training observations
    # ---------------------------------------------------------
    # This is training-data-only behaviour and must happen
    # before the development/test split.

    df_clean = drop_duplicate_rows(
        df_clean,
        config["diagnostics"].get("id_column")
    )

    # ---------------------------------------------------------
    # 4. Separate features / target / audit columns
    # ---------------------------------------------------------

    mnar_sources = (
        config["preprocessing"]
        .get("mnar_indicator_sources", [])
    )

    X, y, extras = split_features_target(
        df_clean,
        config["data"],
        mnar_sources
    )

    # ---------------------------------------------------------
    # 5. Development / locked test split
    # ---------------------------------------------------------

    (
        X_dev,
        X_test,
        y_dev,
        y_test,
        extras_dev,
        extras_test,
    ) = split_dev_test(
        X,
        y,
        extras,
        test_size=config["split"]["test_size"],
        random_state=config["split"]["random_state"],
    )

    # ---------------------------------------------------------
    # 6. Build preprocessing + model pipeline
    # ---------------------------------------------------------

    preprocessor = build_preprocessor(
        config["preprocessing"]
    )

    pipeline = Pipeline([
        (
            "prep",
            preprocessor
        ),
        (
            "model",
            build_model(
                config["model"]
            )
        ),
    ])

    # ---------------------------------------------------------
    # 7. Fit on development data
    # ---------------------------------------------------------

    pipeline.fit(
        X_dev,
        y_dev
    )

    # ---------------------------------------------------------
    # 8. Predictions
    # ---------------------------------------------------------

    y_dev_pred = pipeline.predict(
        X_dev
    )

    y_test_pred = pipeline.predict(
        X_test
    )

    # ---------------------------------------------------------
    # 9. Evaluation
    # ---------------------------------------------------------

    report = evaluate(
        y_dev,
        y_dev_pred,
        y_test,
        y_test_pred
    )

    report += "\n" + fairness_report(
        y_test,
        y_test_pred,
        extras_test,
        sensitive_attr=config["data"]["sensitive_attr"]
    )

    # ---------------------------------------------------------
    # 10. Save results
    # ---------------------------------------------------------

    results_dir = (
        config
        .get("output", {})
        .get("results_dir", "results")
    )

    path = save_run(
        results_dir,
        config,
        report
    )

    print(
        f"Full results saved to {path}"
    )


if __name__ == "__main__":
    main()