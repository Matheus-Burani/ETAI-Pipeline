"""
Preprocessing -- raw data in, model-ready data out: category cleanup, domain-rule and
placeholder -> NaN conversion, training-only de-duplication, fixed-rule engineered features,
the feature/target split,
a leak-safe encoder/scaler preprocessor, and the split that sets the final test set aside.
One file, one obvious place to look for "how does raw data become model-ready".

Two things every function here respects, on purpose:
  - leak-safe: `clean_dataset`, `add_engineered_features` and `split_features_target` learn nothing from the data
    (no means, no category lists, no target), so they are safe to run on any rows.
    Everything that IS learned from data -- imputation, encoding, scaling -- lives inside
    `build_preprocessor`'s ColumnTransformer, which sits inside the model's sklearn
    Pipeline (see `build_pipeline` in model.py). It is therefore re-fit on the training
    part of every CV fold, and never sees the validation fold or the locked test set.
  - deployable from day one: nothing before the split needs the target column. `y` comes
    back as `None` on label-free data to predict, and nothing breaks.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import (
    OneHotEncoder, OrdinalEncoder, TargetEncoder, StandardScaler, MinMaxScaler, RobustScaler,
)
from category_encoders import CountEncoder


def flag_invalid_values(df: pd.DataFrame, rules: dict) -> pd.DataFrame:
    """
    Applies {column: {"min": ..., "max": ...}} domain rules (either bound optional) and sets
    violations to NaN, **in place** on `df`. An "impossible but not missing" value (an age
    of -3, a COMPAS decile score of 15) counts as missing once this runs -- `.isna()` alone
    would never have caught it.

    Returns a small report: the number of violations per column.
    """
    report_rows = []
    for column, bounds in rules.items():
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        lower_ok = numeric >= bounds["min"] if "min" in bounds else pd.Series(True, index=numeric.index)
        upper_ok = numeric <= bounds["max"] if "max" in bounds else pd.Series(True, index=numeric.index)
        violations = numeric.notna() & ~(lower_ok & upper_ok)
        report_rows.append({"column": column, "rule": bounds, "violations": int(violations.sum())})
        df.loc[violations, column] = np.nan
    return pd.DataFrame(report_rows)


def _canonicalize_categories(df: pd.DataFrame, columns_and_maps: dict, placeholder_tokens: set) -> pd.DataFrame:
    out = df.copy()
    for col, mapping in columns_and_maps.items():
        if col not in out.columns:
            continue
        cleaned = out[col].astype(str).str.strip()
        lowered = cleaned.str.lower()
        out[col] = lowered.map(mapping).fillna(cleaned)
        out.loc[out[col].astype(str).str.strip().isin(placeholder_tokens), col] = np.nan
    return out


def clean_dataset(df: pd.DataFrame, diagnostics_config: dict) -> pd.DataFrame:
    """
    Applies the cleaning rules in `diagnostics_config`: placeholder tokens and domain-rule
    violations become NaN, category spellings are canonicalized and redundant columns are
    dropped. Target-agnostic, so it runs unchanged on label-free data to predict.

    Row-preserving: every input row comes out, in the same order. Removing duplicate rows
    is a training-only decision and lives in `drop_duplicate_rows()` -- at prediction time
    every row needs a prediction (e.g. a Kaggle submission needs one per id).
    """
    out = df.copy()
    placeholder_tokens = set(diagnostics_config.get("placeholder_tokens", []))

    # numeric columns that load as text purely because of a placeholder token
    for col in diagnostics_config.get("numeric_text_columns", []):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col].replace(list(placeholder_tokens), np.nan), errors="coerce")

    flag_invalid_values(out, diagnostics_config.get("validity_rules", {}))
    out = _canonicalize_categories(out, diagnostics_config.get("canonical_categories", {}), placeholder_tokens)

    columns_to_drop = [c for c in diagnostics_config.get("redundant_columns", []) if c in out.columns]
    return out.drop(columns=columns_to_drop)


def drop_duplicate_rows(df: pd.DataFrame, id_column: str = None) -> pd.DataFrame:
    """
    TRAINING DATA ONLY. Drops exact duplicate rows and repeated ids (keeping the first), so
    the same person can't be counted twice -- or land in both the development and the
    locked test set. Must run BEFORE `split_dev_test()`.

    Never call this on data you are predicting for: every row there needs a prediction.
    """
    out = df.drop_duplicates()
    if id_column and id_column in out.columns:
        out = out.drop_duplicates(subset=id_column, keep="first")
    return out


# Rows older than this count as "young" (matches the dataset's own "Less than 25" age band).
YOUNG_AGE = 25


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of `df` with three extra numeric columns built from `age` and
    `priors_count`; every other column and every row is left as it was:
      - priors_per_adult_year: priors_count divided by the years lived as an adult
        (age - 17, never below 1). A ratio: 5 priors at 20 is not the same record as
        5 priors at 50.
      - log_priors: log(1 + priors_count). Compresses the long right tail of the counts,
        so the gap between 0 and 2 priors weighs more than the gap between 20 and 22.
      - young_x_priors: priors_count for people younger than YOUNG_AGE, 0 for everyone
        else. An interaction: priors only count for the young group.

    Each value comes from a fixed formula on the same row, so nothing is learned from the
    data (no mean, no quantile, no target). If an input is missing, the new value is NaN.
    """
    out = df.copy()
    adult_years = (out["age"] - 17).clip(lower=1)
    out["priors_per_adult_year"] = out["priors_count"] / adult_years
    out["log_priors"] = np.log1p(out["priors_count"])
    out["young_x_priors"] = out["priors_count"].where(out["age"] < YOUNG_AGE, 0.0).where(out["age"].notna())
    return out


def add_missingness_indicators(df: pd.DataFrame, mnar_indicator_sources: list) -> pd.DataFrame:
    """
    Adds a `<col>_was_missing` flag for each listed (MNAR-diagnosed) column, before that
    column gets imputed -- so a model can still see the missingness pattern even though the
    fill value itself (median/mode) can't carry it. Target-agnostic.
    """
    out = df.copy()
    for col in mnar_indicator_sources:
        if col in out.columns:
            out[f"{col}_was_missing"] = out[col].isna().astype(int)
    return out


def split_features_target(df: pd.DataFrame, data_config: dict, mnar_indicator_sources: list):
    """
    Returns (X, y, extras). `extras` holds the columns kept out of the features for auditing
    (the sensitive attribute and COMPAS's own score). On label-free data `y` is None --
    nothing downstream requires the target column to be present.
    """
    target = data_config["target"]
    sensitive_attr = data_config["sensitive_attr"]
    drop_columns = data_config.get("drop_columns", [])

    df = add_missingness_indicators(df, mnar_indicator_sources)
    y = df[target] if target in df.columns else None

    extras_cols = [c for c in [sensitive_attr, "score_text"] if c in df.columns]
    extras = df[extras_cols].copy() if extras_cols else None

    always_drop = set(drop_columns) | {target, sensitive_attr}
    feature_cols = [c for c in df.columns if c not in always_drop]
    return df[feature_cols], y, extras


_SCALERS = {"none": "passthrough", "standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}

# Each entry takes a random_state (only the target encoder actually uses it).
# Target encoding uses sklearn's TargetEncoder: during fit it *cross-fits* -- each training
# row is encoded with category means computed on the OTHER internal folds, never on its own
# label. Without that, a row's own target leaks into its own feature value, and the model
# learns to trust the encoding more than it deserves. Unseen categories at predict time
# get the overall target mean.
_ENCODERS = {
    "onehot": lambda seed: OneHotEncoder(handle_unknown="ignore", sparse_output=False),
    "ordinal": lambda seed: OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
    "count": lambda seed: CountEncoder(handle_unknown=0, handle_missing=0),
    "target": lambda seed: TargetEncoder(target_type="binary", cv=StratifiedKFold(5, shuffle=True, random_state=seed)),
}


def build_preprocessor(preprocessing_config: dict) -> ColumnTransformer:
    """
    Factory: builds the unfitted, leak-safe ColumnTransformer described by
    `preprocessing_config` (the `preprocessing` section of the config file -- chosen there,
    not hardcoded here): imputation and scaling for numeric columns, imputation and
    encoding for categorical ones, and the missingness flags passed through. Every encoder
    tolerates unseen categories at transform time.

    Nothing is fit here. Fitting happens later, on the training part of each CV fold only,
    because this object is placed inside the model's sklearn Pipeline.
    """
    imputation = preprocessing_config.get("imputation", {})
    scaler_factory = _SCALERS[preprocessing_config["scaler"]]
    scaler = scaler_factory() if callable(scaler_factory) else scaler_factory
    encoder = _ENCODERS[preprocessing_config["encoder"]](preprocessing_config.get("random_state"))

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("numeric_strategy", "median"))),
        ("scale", scaler),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy=imputation.get("categorical_strategy", "most_frequent"))),
        ("encode", encoder),
    ])
    indicator_cols = [f"{c}_was_missing" for c in preprocessing_config.get("mnar_indicator_sources", [])]

    return ColumnTransformer([
        ("numeric", numeric_pipeline, preprocessing_config["numeric_features"]),
        ("categorical", categorical_pipeline, preprocessing_config["categorical_features"]),
        ("indicators", "passthrough", indicator_cols),
    ])


def split_dev_test(X, y, extras, test_size: float, random_state: int):
    """
    Sets the final test set aside. Stratified split of X, y and the extras frame (race and
    score_text, kept for the fairness report) together, so all three stay row-aligned.
    Returns X_dev, X_test, y_dev, y_test, extras_dev, extras_test:
      - development set: everything we are allowed to learn from and compare models on.
        Cross-validation (evaluate.py) splits it again into train/validation folds.
      - locked test set: never used to fit, tune, compare or choose anything, only for the
        final assessment. Its size and seed live in the config file's `test_set` section
        and must never change once chosen.
    """
    return train_test_split(X, y, extras, test_size=test_size, random_state=random_state, stratify=y)