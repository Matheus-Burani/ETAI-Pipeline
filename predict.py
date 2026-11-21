"""
Predicts a CSV of new people with the model saved by `main.py` (no training).

    python predict.py data/compas_new_people.csv predictions.csv
    python predict.py data/compas_new_people.csv predictions.csv --config config_multiclass.yaml

The new file is cleaned with the same rules as the training data, but without
`drop_duplicate_rows()`: every input row gets an answer.
"""
import argparse

import joblib
import pandas as pd
import yaml

from src.preprocessing import clean_dataset, split_features_target


def predict(df: pd.DataFrame, pipeline, config: dict) -> pd.DataFrame:
    """
    One row per input row. Two classes: the id, the probability of class 1 and the prediction
    (at 0.5). Three or more: the id, one probability column per class and the most likely class.
    """
    clean = clean_dataset(df, config["diagnostics"])
    X, _, _ = split_features_target(clean, config["data"], config["preprocessing"]["mnar_indicator_sources"])
    proba = pipeline.predict_proba(X)
    ids = df[config["diagnostics"]["id_column"]].values

    if proba.shape[1] == 2:
        probability = proba[:, 1]
        return pd.DataFrame({"id": ids, "probability": probability,
                             "prediction": (probability >= 0.5).astype(int)})

    names = config["data"].get("class_names") or [str(c) for c in pipeline.classes_]
    out = pd.DataFrame({"id": ids})
    for k, name in enumerate(names):
        out[f"probability_{name}"] = proba[:, k]
    out["prediction"] = [names[k] for k in proba.argmax(axis=1)]
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict a CSV with the model saved by main.py.")
    parser.add_argument("input", help="CSV with the same columns as the training data (no target needed)")
    parser.add_argument("output", nargs="?", default="predictions.csv", help="where to write the predictions")
    parser.add_argument("--config", default="config.yaml", help="the config the model was trained with")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    pipeline = joblib.load(config.get("output", {}).get("model_path", "models/model.joblib"))
    predictions = predict(pd.read_csv(args.input), pipeline, config)
    predictions.to_csv(args.output, index=False)
    print(f"{len(predictions)} rows predicted -> {args.output}")