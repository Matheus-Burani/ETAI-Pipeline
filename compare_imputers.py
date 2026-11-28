"""
Compares the runs in the scoreboard by one setting (the imputer by default, or the anomaly
method), with the corrected paired t-test, and prints a markdown table for the README.

    python compare_imputers.py                          # every run in results/runs.csv
    python compare_imputers.py --model gradient_boosting --baseline median
    python compare_imputers.py --by anomaly --baseline none --tuned true

Run each value with the same model and other settings first (tuning off is fine). If a value
was run more than once, its latest run is used. Runs saved before that column existed are
ignored.
"""
import argparse

import pandas as pd

from src.evaluate import corrected_paired_test

parser = argparse.ArgumentParser(description="Compare runs from the scoreboard by one setting.")
parser.add_argument("--runs", default="results/runs.csv", help="the scoreboard written by main.py")
parser.add_argument("--by", default="imputer", help="the scoreboard column to compare: imputer | anomaly")
parser.add_argument("--model", default=None, help="only runs of this model type")
parser.add_argument("--tuned", choices=["true", "false"], default=None, help="only tuned (or untuned) runs")
parser.add_argument("--baseline", default="median", help="the value the others are compared with (e.g. median, none)")
args = parser.parse_args()

runs = pd.read_csv(args.runs, dtype={"run": str})
if args.by not in runs.columns:
    raise SystemExit(f"{args.runs} has no '{args.by}' column yet: run main.py first")
runs = runs.dropna(subset=[args.by])
if args.model:
    runs = runs[runs["model"] == args.model]
if args.tuned:
    runs = runs[runs["tuned"].astype(str).str.lower() == args.tuned]
runs = runs.sort_values("run").groupby(args.by).tail(1)
folds = [c for c in runs.columns if c.startswith("fold_")]
scores = {row[args.by]: row[folds].astype(float).values for _, row in runs.iterrows()}
if args.baseline not in scores:
    raise SystemExit(f"No run with {args.by} '{args.baseline}' in {args.runs} (found: {sorted(scores)})")
ratio = 1 / (len(folds) - 1)                                 # validation rows / training rows per fold

print(f"| {args.by.capitalize()} | {runs['metric'].iloc[0]}: mean ± std | vs {args.baseline}: difference | folds won | p-value |")
print("|---|---|---|---|---|")
for name, s in scores.items():
    cells = f"| {name} | {s.mean():.3f} ± {s.std(ddof=1):.3f} | "
    if name == args.baseline:
        print(cells + "n/a | n/a | n/a |")
    else:
        t = corrected_paired_test(s, scores[args.baseline], test_train_ratio=ratio)
        print(cells + f"{t['mean_diff']:+.4f} | {t['wins']} | {t['p_value']:.2f} |")