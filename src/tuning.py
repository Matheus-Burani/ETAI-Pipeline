"""
Hyperparameter tuning with Optuna, and nested cross-validation.

Tuning uses the development set only, and scores every candidate by cross-validation of the
whole pipeline on the same folds. The best trial's score is the maximum of many noisy scores,
so it is optimistic; the honest estimate of "tune, then refit" comes from
`nested_cross_validate`.

Preprocessing can be tuned too, because it lives inside the Pipeline: every trial re-fits it
on the training part of each fold, so tuning it is as leak-free as tuning the model. Two kinds
of search-space entries reach it:
  - ordinary Pipeline parameters, e.g. `prep__numeric__impute__strategy` (a string option),
    which are passed to `set_params` as they are;
  - the two *choices of object*, `prep__numeric__scale` (scaler) and
    `prep__categorical__encode` (encoder). A YAML file can only hold names, so these are
    written as categorical choices of names ("robust", "standard", "onehot", "target", ...)
    and turned into fresh scaler/encoder objects by `_resolve_params`.
"""
import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import cross_val_score

from src.preprocessing import _SCALERS, _ENCODERS

SCALER_KEY = "prep__numeric__scale"
ENCODER_KEY = "prep__categorical__encode"


def _resolve_params(params: dict, random_state: int) -> dict:
    """
    Turns the names chosen for the scaler and the encoder into real objects, using the same
    registries as `build_preprocessor`; every other parameter is passed through unchanged.
    A new object is built on every call, so trials never share a fitted scaler or encoder.
    """
    out = dict(params)
    if SCALER_KEY in out:
        factory = _SCALERS[out[SCALER_KEY]]
        out[SCALER_KEY] = factory() if callable(factory) else factory      # "none" -> "passthrough"
    if ENCODER_KEY in out:
        out[ENCODER_KEY] = _ENCODERS[out[ENCODER_KEY]](random_state)
    return out


def tune_pipeline(pipeline, X, y, cv, scoring: str, search_space: dict, n_trials: int,
                  random_state: int, n_jobs: int = 1):
    """
    Runs an Optuna study (seeded TPE sampler) on (X, y). Each trial draws one value per entry
    of `search_space` (`type`: int / float / categorical; keys are Pipeline parameter names,
    e.g. `model__max_depth` or `prep__numeric__impute__strategy`), sets them on a fresh copy
    of `pipeline`, and returns its mean CV score over `cv`.

    Returns (best_pipeline, study): an unfitted copy of `pipeline` with the best trial's
    hyperparameters, and the study with every trial (mean score and fold std).
    """
    def objective(trial):
        params = {}
        for name, spec in search_space.items():
            if spec["type"] == "int":
                params[name] = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "float":
                params[name] = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
            elif spec["type"] == "categorical":
                params[name] = trial.suggest_categorical(name, spec["choices"])
            else:
                raise ValueError(f"Unknown search-space type for {name}: {spec['type']}. "
                                 "Options: int, float, categorical")
        candidate = clone(pipeline).set_params(**_resolve_params(params, random_state))
        scores = cross_val_score(candidate, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
        trial.set_user_attr("std", float(scores.std(ddof=1)))
        return scores.mean()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=random_state))
    study.optimize(objective, n_trials=n_trials)
    best_pipeline = clone(pipeline).set_params(**_resolve_params(study.best_params, random_state))
    return best_pipeline, study


def nested_cross_validate(pipeline, X, y, outer_cv, inner_cv, scoring: str, search_space: dict,
                          n_trials: int, random_state: int, n_jobs: int = 1):
    """
    Honest estimate of the tuning procedure. For every outer fold: tune on the outer-training
    rows (`tune_pipeline` with `inner_cv`), refit the best pipeline on them, and score it on
    the outer-validation rows, which took no part in the tuning.

    Returns (fold_scores, y_oof), as `cross_validate_pipeline` does, with two extra kinds of
    columns: `inner_best` (the tuning score inside that fold) and the hyperparameters chosen.
    """
    scorer = get_scorer(scoring)
    rows = []
    y_oof = np.empty(len(X), dtype=np.asarray(y).dtype)
    for fold, (train_idx, val_idx) in enumerate(outer_cv.split(X, y), start=1):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        best_pipeline, study = tune_pipeline(pipeline, X_train, y_train, inner_cv, scoring,
                                             search_space, n_trials, random_state, n_jobs)
        best_pipeline.fit(X_train, y_train)

        train_score = scorer(best_pipeline, X_train, y_train)
        val_score = scorer(best_pipeline, X_val, y_val)
        y_oof[val_idx] = best_pipeline.predict(X_val)
        rows.append({"fold": fold, "train": train_score, "validation": val_score,
                     "gap": train_score - val_score, "inner_best": study.best_value,
                     **study.best_params})
    return pd.DataFrame(rows), y_oof


def tuning_report(study, nested_scores: pd.DataFrame, scoring: str = "accuracy", top: int = 5) -> str:
    """
    Best hyperparameters and top trials of the final study, plus the optimism check (mean
    tuning score vs mean honest score over the outer folds), as text (printed and returned).
    """
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs"))
    trials = trials.rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
    trials = trials.rename(columns={"number": "trial", "value": f"mean {scoring}"})
    inner, outer = nested_scores["inner_best"].mean(), nested_scores["validation"].mean()
    lines = [
        f"Tuning on the whole development set ({len(study.trials)} Optuna trials, metric: {scoring})",
        f"Best hyperparameters: {study.best_params}",
        f"Best mean CV score:   {study.best_value:.3f}  (the maximum over all trials -- optimistic)",
        "",
        f"Top {top} trials:",
        trials.sort_values(f"mean {scoring}", ascending=False).head(top)
              .to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        "Optimism check (nested CV):",
        f"  tuning score inside the outer folds (inner_best) mean = {inner:.3f}",
        f"  honest score on the outer folds (validation)     mean = {outer:.3f}",
        f"  optimism = {inner - outer:+.3f}   -> report the honest one",
    ]
    text = "\n".join(lines)
    print(text)
    return text