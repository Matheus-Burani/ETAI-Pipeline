"""
Do the most unusual people reoffend more often? A look at the data, not a model evaluation.

    python check_anomalies.py                       # top 5% by score, config.yaml
    python check_anomalies.py --top 0.10

On the development rows only (the test set stays locked): fits the preprocessor and each
anomaly detector (`AnomalyScore`, Isolation Forest and LOF), takes the `--top` fraction of
people with the highest score and compares how often they reoffend with everyone else.
"""
import argparse

import numpy as np
import pandas as pd
import yaml

from src.data import load_data
from src.preprocessing import (AnomalyScore, build_preprocessor, clean_dataset, drop_duplicate_rows,
                               split_dev_test, split_features_target)

parser = argparse.ArgumentParser(description="Reoffending rate of the most unusual people.")
parser.add_argument("--config", default="config.yaml")
parser.add_argument("--top", type=float, default=0.05, help="fraction of people with the highest score")
args = parser.parse_args()

with open(args.config) as f:
    config = yaml.safe_load(f)

df = drop_duplicate_rows(clean_dataset(load_data(config["data"]["path"]), config["diagnostics"]),
                         config["diagnostics"].get("id_column"))
X, y, extras = split_features_target(df, config["data"], config["preprocessing"].get("mnar_indicator_sources", []))
X_dev, _, y_dev, _, _, _ = split_dev_test(X, y, extras, config["test_set"]["size"], config["test_set"]["random_state"])
y_dev = np.asarray(y_dev)

Z = build_preprocessor(config["preprocessing"]).fit_transform(X_dev, y_dev)
anomaly = config.get("anomaly") or {}

rows = []
for method in ["isolation_forest", "lof"]:
    step = AnomalyScore(method=method, n_neighbors=anomaly.get("n_neighbors", 20),
                        random_state=anomaly.get("random_state", 42))
    score = step.fit_transform(Z)[:, -1]
    top = score >= np.quantile(score, 1 - args.top)
    rows.append({"method": method, "people in top": int(top.sum()),
                 "reoffend rate, top": y_dev[top].mean(), "reoffend rate, rest": y_dev[~top].mean(),
                 "rank corr. score vs reoffended": pd.Series(score).corr(pd.Series(y_dev), method="spearman")})

print(f"Development rows: {len(y_dev)}   |   overall reoffend rate: {y_dev.mean():.3f}   |   top {args.top:.0%} by score\n")
print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
