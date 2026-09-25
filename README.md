# Matheus Burani 20260577

# Baseline Predictive Pipeline -- ETAI


The task: predict two-year recidivism using ProPublica's COMPAS
dataset -- the data behind a real 2016 investigation into a risk-
assessment algorithm actually used by US courts to help inform bail and sentencing decisions. See `data/README.md` for the full problem description and a complete data dictionary before you start.


## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress


| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (count encoding + robust scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. |

## Environment setup

### macOS / Linux
```bash
python3 -m venv venv                 
source venv/bin/activate              
pip install -r requirements.txt     
```

### Windows -- PowerShell
```powershell
python -m venv venv                 
venv\Scripts\activate                
pip install -r requirements.txt 
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Windows -- cmd.exe
Same three steps as above, just with cmd's own activation command:
```cmd
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**macOS / Linux**
```bash
source venv/bin/activate
python main.py
```

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.

# Matheus Burani 20260577

## week 2 improvements and results

We can see that in Week 2, the logistic regression model has a train accuracy of 0.679 and a test accuracy of 0.680, essentially no difference between the two. The decision tree (max_depth=5, min_samples_leaf=25, class_weight="balanced") has a train accuracy of   0.677 and a test accuracy of 0.667, a gap of about 0.010. Both gaps are small, which means that with these constraints the decision tree is not memorizing the training data, it is generalizing about as well as the logistic regression model, not worse.

It is possible to observe that the results for precision, recall and F1 score (for the recidivism/positive class) were respectively:

0.62, 0.67, 0.65 – decision tree
0.66, 0.60, 0.63 – logistic regressionn

Here we can notice a trade-off rather than one model being strictly better: logistic regression has higher precision, meaning that when it predicts recidivism it is right more often, while the decision tree has higher recall, meaning it catches more of the people who actually do reoffend.

## week 3 improvements and results

### What changed between the week 2 and week 3? 

Missing values: Week 2 drops any row with a missing value. Week 3 imputes instead (median for numeric, most-frequent for categorical) and adds a _was_missing flag so the model can still see the missingness pattern.

Categorical encoding: Week 2 blindly one-hot encodes with pd.get_dummies — no cleanup, which is why race ended up fragmented into 16 messy variants. Week 3 first canonicalizes categories (fixing casing/spelling, converting placeholders to NaN) and then encodes with a configurable, unseen-category-safe method (one-hot/ordinal/count/target).

Data quality: Week 2 has no checks beyond dropping NaNs. Week 3 adds validity-rule checks, de-duplication, and redundant-column removal.

Leakage (the big one): Week 2 encodes the entire dataset before splitting into train/test, so test-set information leaks into the encoding. Week 3 fixes this by making use of imputation, encoding, and scaling all fit only on the training split, via a proper Pipeline/ColumnTransformer.

Scaling: Week 2 does none. Week 3 adds optional standard/minmax/robust scaling.

Unchanged: Both still use a single stratified split (no cross-validation), and both keep race out of the model's actual input features, using it only for the fairness audit.

### Results

In Week 3, after the preprocessing changes, the picture shifts. Logistic regression now has a train accuracy of 0.676 and a test accuracy of 0.657, a gap of +0.019 — noticeably larger than its Week 2 gap, and for the first time showing a bit of overfitting rather than a clean generalization. The decision tree, meanwhile, has a train accuracy of 0.669 and a test accuracy of 0.663, a gap of +0.006 — actually tighter than its own Week 2 gap. So the preprocessing change pushed logistic regression toward overfitting while the decision tree became, if anything, slightly more stable.

0.62, 0.67, 0.64 – decision tree
0.66, 0.48, 0.56 – logistic regression

Here the decision tree barely moves at all between weeks — precision, recall, and F1 stay essentially flat, which lines up with its stable train/test gap. Logistic regression tells a different story: precision holds steady at 0.66, but recall drops sharply from 0.60 to 0.48, dragging F1 down from 0.63 to 0.56. In other words, the preprocessing change made logistic regression considerably more conservative — it's still right just as often when it does predict recidivism, but it's now missing far more of the people who actually reoffend.

Taken together, the trade-off that existed within Week 2 (logistic regression favoring precision, the decision tree favoring recall) still holds in Week 3, but it's become more lopsided: the decision tree kept its balance across both weeks, while logistic regression lost a substantial chunk of its recall, making the gap between the two models wider than it was before.
