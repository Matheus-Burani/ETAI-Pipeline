"""
Evaluation on the development set: stratified k-fold cross-validation of the whole pipeline
on several metrics, out-of-fold probabilities and reports, a fairness check and a paired
comparison of two models. The locked test set is never used here.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score, get_scorer, make_scorer
from sklearn.model_selection import cross_validate


def build_scorers(metrics: list) -> dict:
    """
    {name: scorer} for scikit-learn metric names (`accuracy`, `roc_auc`, `f1_macro`, `roc_auc_ovr`, ...),
    plus `fbeta_<beta>` (e.g. `fbeta_2`: recall weighted twice as much as precision, two classes)
    and `fbeta_macro_<beta>` (the same, averaged over the classes, for more than two).
    """
    scorers = {}
    for name in metrics:
        if name.startswith("fbeta_macro_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.rsplit("_", 1)[1]), average="macro",
                                        zero_division=0)
        elif name.startswith("fbeta_"):
            scorers[name] = make_scorer(fbeta_score, beta=float(name.split("_", 1)[1]), zero_division=0)
        else:
            scorers[name] = get_scorer(name)
    return scorers


def report_proba(model, X) -> np.ndarray:
    """
    Probabilities in the shape the reports use: P(class 1) as one number per row when there
    are two classes (a threshold can be moved on it), the whole rows x classes matrix when
    there are more (the prediction is the most likely class).
    """
    proba = model.predict_proba(X)
    return proba[:, 1] if proba.shape[1] == 2 else proba


def cross_validate_pipeline(pipeline, X, y, cv, scorers: dict, selection_metric: str, n_jobs: int = 1):
    """
    Fits a fresh copy of `pipeline` (preprocessing included) on each fold's training part and
    scores it on that fold's validation part, on every metric in `scorers`.

    Returns (fold_scores, oof_proba):
      - fold_scores: one row per fold with the validation score of every metric, plus the
        train score and train-validation gap of `selection_metric`;
      - oof_proba: out-of-fold probabilities for every row, from the fold model that did not
        train on it: P(class 1) for two classes, one column per class for more (see
        `report_proba`). `cv` must be a partition (each row validated exactly once).
    """
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scorers, return_train_score=True,
                            return_estimator=True, return_indices=True, n_jobs=n_jobs)
    fold_scores = pd.DataFrame({"fold": range(1, cv.get_n_splits() + 1)})
    fold_scores["train"] = scores[f"train_{selection_metric}"]
    fold_scores["validation"] = scores[f"test_{selection_metric}"]
    fold_scores["gap"] = fold_scores["train"] - fold_scores["validation"]
    for name in scorers:
        fold_scores[name] = scores[f"test_{name}"]

    oof_proba = None
    for model, val_idx in zip(scores["estimator"], scores["indices"]["test"]):
        proba = report_proba(model, X.iloc[val_idx])
        if oof_proba is None:
            oof_proba = np.empty((len(X),) + proba.shape[1:])
        oof_proba[val_idx] = proba
    return fold_scores, oof_proba


def cv_report(fold_scores: pd.DataFrame, metrics: list, selection_metric: str, title: str) -> str:
    """Per-fold table and the mean and std of every metric, as text."""
    lines = [
        f"{title} ({len(fold_scores)} stratified folds, selection metric: {selection_metric})",
        "",
        fold_scores.to_string(index=False, float_format=lambda v: f"{v:.3f}"),
        "",
        fold_scores[metrics].agg(["mean", "std"]).T.to_string(float_format=lambda v: f"{v:.3f}"),
        f"\n{selection_metric}: train-validation gap mean = {fold_scores['gap'].mean():+.3f}",
    ]
    return "\n".join(lines)


def corrected_paired_test(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Corrected paired t-test (Nadeau & Bengio, 2003) for two models scored on the same CV folds.
    Folds share most of their training rows, so their scores are correlated and the plain
    t-test is overconfident; the variance of the differences is inflated by `test_train_ratio`
    (validation rows / training rows per fold: 1 / (k - 1) for k-fold).

    Returns the mean difference (a - b), the folds where a scores higher, t and the two-sided
    p-value.
    """
    diff = np.asarray(scores_a, dtype=float) - np.asarray(scores_b, dtype=float)
    k = len(diff)
    se = np.sqrt((1 / k + test_train_ratio) * diff.var(ddof=1))
    t = float(diff.mean() / se) if se > 0 else 0.0
    p_value = float(2 * stats.t.sf(abs(t), df=k - 1))
    return {"mean_diff": float(diff.mean()), "wins": f"{int((diff > 0).sum())}/{k}", "t": t, "p_value": p_value}


def oof_classification_report(y_true, oof_proba, threshold: float = 0.5, class_names: list = None) -> str:
    """
    Confusion matrix and classification report of the out-of-fold predictions, as text. Two
    classes: class 1 when P(1) >= `threshold`. More classes: the most likely class (no
    threshold). `class_names` (in label order) names the classes in the multiclass tables.
    """
    if oof_proba.ndim == 1:
        y_pred = (oof_proba >= threshold).astype(int)
        cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=["true 0", "true 1"],
                          columns=["predicted 0", "predicted 1"])
        return (f"Out-of-fold predictions, development set (threshold {threshold}):\n\n"
                f"{cm.to_string()}\n\n{classification_report(y_true, y_pred, zero_division=0)}")

    classes = np.unique(y_true)
    names = list(class_names) if class_names else [str(c) for c in classes]
    y_pred = classes[oof_proba.argmax(axis=1)]
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes),
                      index=[f"true {n}" for n in names], columns=[f"predicted {n}" for n in names])
    return (f"Out-of-fold predictions, development set (most likely class):\n\n"
            f"{cm.to_string()}\n\n"
            f"{classification_report(y_true, y_pred, labels=classes, target_names=names, zero_division=0)}")


def fairness_report(y_true, oof_proba, extras: pd.DataFrame, sensitive_attr: str = "race",
                    threshold: float = 0.5, class_names: list = None) -> str:
    """
    False positive rate (share of people who did not reoffend but were predicted to) per
    group of `sensitive_attr`, for the model's out-of-fold predictions and for COMPAS's own
    score (`score_text` other than "Low" counts as predicted to reoffend), on the same rows.
    Rows with an unknown group are reported as "unknown". A simple check, not a full audit.

    With more than two classes the same idea is applied to the highest class (the last one):
    the share of people who were not truly in it but were predicted to be, per group, for the
    model only (here COMPAS's own score is the target, so there is nothing to compare with).
    """
    if oof_proba.ndim == 2:
        return _multiclass_fairness_report(y_true, oof_proba, extras, sensitive_attr, class_names)
    df = extras.copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["y_pred_model"] = (oof_proba >= threshold).astype(int)
    df["y_pred_compas"] = (df["score_text"] != "Low").astype(int)

    lines = [
        f"False positive rate by {sensitive_attr} (development set, out-of-fold)",
        "(share of people who did NOT reoffend, but were predicted to)",
        "",
    ]
    for label, col in [("Our model", "y_pred_model"), ("COMPAS's own score", "y_pred_compas")]:
        lines.append(f"  {label}:")
        for group, g in df.groupby(sensitive_attr):
            negatives = g[g["y_true"] == 0]
            if len(negatives) == 0:
                continue
            fpr = (negatives[col] == 1).mean()
            lines.append(f"    {group:<20s} FPR = {fpr:.2f}  (n={len(negatives)})")
        lines.append("")
    return "\n".join(lines)


def _multiclass_fairness_report(y_true, oof_proba, extras, sensitive_attr, class_names) -> str:
    """False-positive rate of the highest class, per group of `sensitive_attr` (model only)."""
    classes = np.unique(y_true)
    top = classes[-1]
    top_name = class_names[-1] if class_names else str(top)
    df = extras[[sensitive_attr]].copy()
    df[sensitive_attr] = df[sensitive_attr].astype(object).fillna("unknown")
    df["y_true"] = pd.Series(y_true).values
    df["predicted_top"] = classes[oof_proba.argmax(axis=1)] == top

    lines = [
        f"False {top_name} rate by {sensitive_attr} (development set, out-of-fold)",
        f"(share of people who were NOT truly {top_name}, but were predicted {top_name})",
        "",
        "  Our model:",
    ]
    for group, g in df.groupby(sensitive_attr):
        negatives = g[g["y_true"] != top]
        if len(negatives) == 0:
            continue
        lines.append(f"    {group:<20s} rate = {negatives['predicted_top'].mean():.2f}  (n={len(negatives)})")
    lines.append("")
    return "\n".join(lines)