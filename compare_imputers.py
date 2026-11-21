"""
Compares the runs in the scoreboard by imputer, with the corrected paired t-test, and prints
a markdown table for the README.

    python compare_imputers.py                          # every run in results/runs.csv
    python compare_imputers.py --model gradient_boosting --baseline median

Run each imputer with the same model and settings first (tuning off is fine). If an imputer
was run more than once, its latest run is used. Runs saved before the `imputer` column
existed are ignored.
"""
import argparse

import pandas as pd

from src.evaluate import corrected_paired_test

parser = argparse.ArgumentParser(description="Compare imputers from the scoreboard.")
parser.add_argument("--runs", default="results/runs.csv", help="the scoreboard written by main.py")
parser.add_argument("--model", default=None, help="only runs of this model type")
parser.add_argument("--baseline", default="median", help="the imputer the others are compared with")
args = parser.parse_args()

runs = pd.read_csv(args.runs, dtype={"run": str}).dropna(subset=["imputer"])
if args.model:
    runs = runs[runs["model"] == args.model]
runs = runs.sort_values("run").groupby("imputer").tail(1)
folds = [c for c in runs.columns if c.startswith("fold_")]
scores = {row["imputer"]: row[folds].astype(float).values for _, row in runs.iterrows()}
if args.baseline not in scores:
    raise SystemExit(f"No run with imputer '{args.baseline}' in {args.runs} (found: {sorted(scores)})")
ratio = 1 / (len(folds) - 1)                                 # validation rows / training rows per fold

print(f"| Imputer | {runs['metric'].iloc[0]}: mean ± std | vs {args.baseline}: difference | folds won | p-value |")
print("|---|---|---|---|---|")
for name, s in scores.items():
    cells = f"| {name} | {s.mean():.3f} ± {s.std(ddof=1):.3f} | "
    if name == args.baseline:
        print(cells + "n/a | n/a | n/a |")
    else:
        t = corrected_paired_test(s, scores[args.baseline], test_train_ratio=ratio)
        print(cells + f"{t['mean_diff']:+.4f} | {t['wins']} | {t['p_value']:.2f} |")