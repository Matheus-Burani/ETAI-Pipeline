"""
Entry point for the predictive pipeline.

Run with:
    python main.py

This orchestrates the full pipeline:
    load config -> load data -> diagnose/clean -> drop duplicate rows (training only)
    -> split features/target
    -> set the final test set aside, locked
    -> stratified k-fold cross-validation of preprocessing + model on the development set
    -> [if tuning is enabled: nested CV of the tuning procedure, then tuning on all development rows]
    -> out-of-fold classification report + fairness check
    -> refit the final model on the whole development set -> save results
"""
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_dev_test
from src.model import build_pipeline
from src.evaluate import cross_validate_pipeline, cv_report, oof_classification_report, fairness_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # Read the tuning settings first, so a missing search space fails immediately instead of
    # after the baseline cross-validation has already run. `or {}` also covers an empty
    # `tuning:` key in the YAML, which loads as None.
    tuning_config = config.get("tuning") or {}
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")

    # load + diagnose-and-clean: nothing here is learned from the data, so it's safe to run
    # on the whole dataset -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])          # row-preserving: also safe for new data
    # training data only: the same person must not count twice, or sit in both dev and test
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # The final test set is set aside HERE and never used again in this script. Every
    # decision from now on (preprocessing, model, hyperparameters) is made on the
    # development set only. The test set is used only for the final assessment.
    X_dev, X_test, y_dev, y_test, extras_dev, extras_test = split_dev_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    # preprocessing lives INSIDE the pipeline, so cross-validation re-fits it on the
    # training part of every fold -- the validation fold never leaks into its own preprocessing
    pipeline = build_pipeline(config["preprocessing"], config["model"])

    # a fixed random_state = the same folds on every run and for every model, so comparing
    # two models' fold scores is a like-for-like (paired) comparison
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # Baseline: fold scores + out-of-fold predictions (each row predicted by the fold model
    # that did NOT train on it), using the hyperparameters written in config.yaml.
    fold_scores, y_oof = cross_validate_pipeline(pipeline, X_dev, y_dev, cv, scoring, n_jobs=n_jobs)
    header = f"Hyperparameters from config.yaml: {config['model'].get('params')}"
    print(header)
    report = header + "\n" + cv_report(fold_scores, scoring)

    if tuning_enabled:
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)       # one log line per trial is too noisy
        # A fixed seed keeps the inner folds identical for every trial, so candidates are
        # compared on the same data.
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # Honest estimate of "tune, then fit", on the same outer folds as the baseline. Tuning
        # happens inside each outer-training part, so the outer-validation rows never
        # influence the choice of hyperparameters. Its out-of-fold predictions replace the
        # baseline ones, so the reports below describe the procedure we actually ship.
        print(f"\nNested cross-validation: {cv.get_n_splits()} outer folds x {n_trials} trials x "
              f"{inner_cv.get_n_splits()} inner folds ...")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_dev, y_dev, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n\nNested cross-validation (the tuning procedure, estimated honestly):\n"
        report += cv_report(nested_scores, scoring)

        # The hyperparameters that are kept: the same tuning procedure, run once on ALL
        # development rows. `pipeline` becomes an unfitted copy with the best trial's values.
        # Its own best score is optimistic (a maximum over noisy trials), so it is not reported
        # as the model's performance -- the nested score above is.
        pipeline, study = tune_pipeline(
            pipeline, X_dev, y_dev, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        print()
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_dev, y_oof)
    report += "\n" + fairness_report(y_dev, y_oof, extras_dev, sensitive_attr=config["data"]["sensitive_attr"])

    # the model we'd actually use: same pipeline (tuned, if tuning ran), refit on EVERY
    # development row. CV above estimated how well this recipe does; it didn't produce a model.
    pipeline.fit(X_dev, y_dev)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_dev)} development rows."
    if tuning_enabled:
        refit += f" Tuned hyperparameters: {study.best_params}"
    print(refit)
    report += "\n" + refit + "\n"

    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_dev)} rows.")
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()