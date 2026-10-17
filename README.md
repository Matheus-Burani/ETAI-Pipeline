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
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see "Ensembles" below |
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

## Week 2

We can see that in Week 2, the logistic regression model has a train accuracy of 0.679 and a test accuracy of 0.680, essentially no difference between the two. The decision tree (max_depth=5, min_samples_leaf=25, class_weight="balanced") has a train accuracy of   0.677 and a test accuracy of 0.667, a gap of about 0.010. Both gaps are small, which means that with these constraints the decision tree is not memorizing the training data, it is generalizing about as well as the logistic regression model, not worse.

It is possible to observe that the results for precision, recall and F1 score (for the recidivism/positive class) were respectively:

0.62, 0.67, 0.65 – decision tree
0.66, 0.60, 0.63 – logistic regressionn

Here we can notice a trade-off rather than one model being strictly better: logistic regression has higher precision, meaning that when it predicts recidivism it is right more often, while the decision tree has higher recall, meaning it catches more of the people who actually do reoffend.

## Week 3

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

## Week 4

Prepareded the preprocessing for the introduction of cross-validating methods and introduced the Random Forests as an algorithm for the model


### Preprocessing decisions

| Column(s) | Issue found | Mechanism | What was done |
|---|---|---|---|
| `age` | 2.0% missing | MCAR | median impute, no indicator needed |
| `juv_fel_count` | 3.0% missing | MCAR | median impute, no indicator needed |
| `priors_count` | ~7% missing (incl. placeholder tokens) | MNAR -- tied to `age_cat` | median impute + `priors_count_was_missing` flag |
| `c_charge_degree` | 3.2% missing | MNAR -- tied to `age_cat` | mode impute + `c_charge_degree_was_missing` flag |
| `race` | ~1% missing (placeholder tokens) | MCAR | mode impute, no indicator (excluded from model features anyway) |
| `sex` | ~1.5% missing (incl. placeholder tokens) | MCAR | mode impute, no indicator needed |
| `age`, `decile_score`, `juv_fel_count`, `priors_count` | invalid values (out-of-range or negative) | domain rule | converted to `NaN` before imputation |
| `sex` / `race` / `c_charge_degree` / `score_text` | inconsistent category spelling (casing, whitespace, abbreviations) | data entry | canonicalized to one spelling per category |
| whole rows | 72 exact-duplicate rows, all sharing a repeated `id` | data entry | dropped, kept first occurrence |
| `prior_offenses`, `age_in_months`, `juvenile_total` | redundant with other columns (correlation r=1.00, or -- for `juvenile_total` -- an exact sum caught only by VIF) | multicollinearity | dropped |

**Encoder/scaler pair:** chosen by hand in `config.yaml` -- **target encoding** (compact, informative and **robust scaling** (median/IQR, so the few extreme counts don't set the scale). The alternatives (`onehot`/`ordinal`/`count`, `none`/`standard`/`minmax`) are one config change away.

### Results

Model performance

In Week 4, the evaluation changes from a single 80/20 split to 5-fold stratified cross-validation on the development set, so the numbers below are validation averages rather than test scores. Logistic regression now has a train accuracy of 0.675 and a validation accuracy of 0.672, a gap of +0.003, much tighter than its +0.019 gap in Week 3, so the small overfitting signal from last week disappears. The decision tree has a train accuracy of 0.672 and a validation accuracy of 0.661, a gap of +0.012, slightly larger than its +0.006 in Week 3 but still small. The ranking between the two flips (the tree was ahead by 0.006 in Week 3, logistic regression is ahead by 0.011 now), but both differences are within the noise, so the two models are effectively tied on accuracy. The new random forest is the clear outlier: it reaches 0.733 on train but only 0.650 on validation, a gap of +0.083 that shows up on every fold, which means it is memorizing the training data.

Precision, recall and F1 for class 1 (reoffended), Week 4:

0.62, 0.65, 0.63 – decision tree
0.68, 0.51, 0.58 – logistic regression
0.62, 0.58, 0.60 – random forest

The decision tree barely moves between weeks (recall 0.67 to 0.65, F1 0.64 to 0.63), which matches its stable gap. Logistic regression is also close to Week 3, with precision rising slightly from 0.66 to 0.68 and recall from 0.48 to 0.51, so F1 moves from 0.56 to 0.58. It is still the conservative model: it is right more often when it predicts recidivism, but it misses about half of the people who actually reoffend. The random forest sits in between on recall but is weaker than both on F1, so it offers no advantage here.

On fairness, the picture is stable. The false positive rate for African-American versus Caucasian defendants is 0.26 vs 0.13 for logistic regression (2.0x, the same as Week 3), 0.41 vs 0.26 for the decision tree (1.58x, down slightly from 1.73x), and 0.36 vs 0.23 for the random forest (1.57x). COMPAS is at 0.45 vs 0.23 (1.96x). Logistic regression has the lowest absolute FPRs, but only because it predicts "reoffend" less often, and its ratio is as bad as COMPAS's. The tree's small improvement in ratio is not large enough to call a real change. The Hispanic and Other groups moved between weeks mainly because Week 3 had only 54 to 85 people per group, and the Week 4 groups (311 and 185) are more reliable. The Asian (n=21) and Native American (n=6) figures are still too small to interpret.

Taken together, the Week 3 trade-off still holds in Week 4: logistic regression favors precision and the decision tree favors recall. What changed is mostly the quality of the evidence, since cross-validation on four times as much data shows that the two models are tied on accuracy at about 0.66 and that neither reduces the roughly 2x disparity in false positives. The random forest is the only model that clearly loses, because of overfitting. One caveat for the write-up is that the Week 3 test split is the same one Week 4 labels as locked, so it is no longer completely untouched for logistic regression and the tree.

## Week 5
### Improvements - Hyperparameter tuning 

1. **Nested cross-validation** -- for each of the 5 outer CV folds, runs a full Optuna study on the outer-training rows only (each trial = 5-fold CV of the whole pipeline), refits the best candidate on those rows and scores it on the outer-validation fold. This is the honest estimate of the tuned model.
2. **Tuning on the whole development set** -- the same study, once, on every development row: these are the hyperparameters the final model uses.

The best trial's score is printed too, but it is **optimistic** (the maximum of many noisy CV scores) -- report the nested one. Search spaces live in `config.yaml` under `tuning.search_spaces`, one per model type, keyed by Pipeline parameter name (`model__max_depth`); a model without a search space can't be tuned until you add one. Cost: `(cv.n_splits + 1) x n_trials x tuning.n_splits` fits.

### Results

The dummy baseline predicts “did not reoffend” for everyone and scores 0.549 accuracy in every fold. That is just the share of class 0 in the development set (3,170 of 5,771), so any real model has to beat 0.549. It never flags anyone, so its class 1 precision, recall and F1 are all 0.00, and its macro F1 is only 0.35.

This week’s models are all clearly above that floor, but the amount depends on whether they are tuned. With only the default config, logistic regression reaches 0.672 validation accuracy (+0.123 over the dummy), the random forest 0.647 (+0.098) and the decision tree 0.610 (+0.061). The unconstrained tree and forest both show a train/validation gap of +0.085, which means they memorize the training data. The tree loses most of its advantage over the dummy this way. After tuning, measured with nested CV, which is the honest estimate, all three models land close together:

0.678, gap +0.002 – logistic regression (C≈0.0025, balanced)
0.678, gap +0.010 – random forest (max_depth 22, min_samples_leaf 56, max_features 0.5)
0.672, gap +0.013 – decision tree (max_depth 6, min_samples_leaf 101, gini)

That is about +0.12 to +0.13 over the dummy for every model. Tuning helps the tree (0.610 to 0.672) and forest (0.647 to 0.678) a lot because it removes the overfitting. It barely changes logistic regression (0.672 to 0.678), which was already well regularized. The three tuned scores sit within one fold-level standard deviation (0.015 to 0.017), so there is no meaningful ranking. The optimism check is small for all of them (+0.002 to +0.004), so the tuning is not inflating the results. The top Optuna trials are also nearly identical (about 0.677 to 0.681), so the tuning surface is flat.

Precision, recall and F1 for class 1 (reoffended), from the out-of-fold predictions (their accuracies match the nested CV scores, so I read them as the tuned models):

0.00, 0.00, 0.00 – dummy
0.67, 0.54, 0.60 – decision tree
0.67, 0.57, 0.61 – random forest
0.67, 0.57, 0.61 – logistic regression

This is where the baseline matters most. The dummy gets 55% accuracy while finding none of the people who reoffend, and the models find 54% to 57% of them at 67% precision. The three models are nearly identical, with the tree slightly lower on recall. Compared with Week 4, the logistic regression and the tree are less different in their error profile now. The balanced tree’s recall of 0.65 has dropped to 0.54, because the tuned tree no longer uses class_weight='balanced'. Logistic regression’s recall rose from 0.51 to 0.57 because tuning selected class_weight='balanced'.

On fairness, the dummy’s false positive rate is 0.00 for every group. This is trivially “fair” because it never predicts reoffending, which shows that FPR is only meaningful together with recall. For the models, the false positive rate for African-American versus Caucasian defendants is 0.29 vs 0.17 for the tree (1.71x), 0.30 vs 0.17 for the forest (1.76x), and 0.31 vs 0.18 for logistic regression (1.72x). COMPAS is at 0.45 vs 0.23 (1.96x). All three models have lower absolute FPRs than COMPAS, and a somewhat smaller ratio. But all still flag non-reoffending African-American defendants at about 1.7 times the rate of Caucasian ones, so no model removes the disparity. The Asian (n=21), Native American (n=6) and new “unknown” (n=66) groups are too small to interpret.

Taken together, the baseline confirms the models learn real signal (about +0.13 over predicting the majority class), but also that the ceiling is low. All three tuned models plateau around 0.67 to 0.68 and a tuned logistic regression with very strong regularization (C≈0.0025) matches a random forest, which suggests the features carry limited signal. So the simplest model is the best choice by default: logistic regression ties the forest on accuracy, F1 and fairness, and has the smallest train/validation gap. One caveat is that the tuned numbers come from the nested CV procedure, while the final models are refit on the full development set with the Optuna-selected hyperparameters, whose best CV scores (0.679 to 0.681) are optimistic.

## week 6

### Ensembles (without feature engen)

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.672 ± 0.013 (+0.003) | 0.671 ± 0.010 (+0.009) | 0.679 | C=0.019, class_weight=None, impute=mean, scaler=none, encoder=target |
| Decision tree | 0.610 ± 0.018 (+0.085) | 0.674 ± 0.013 (+0.011) | 0.680 | max_depth=15, min_samples_leaf=94, criterion=entropy, impute=median, encoder=target |
| Random forest | 0.650 ± 0.018 (+0.083) | 0.676 ± 0.012 (+0.012) | 0.682 | max_depth=4, min_samples_leaf=8, max_features=sqrt, class_weight=None, impute=median, encoder=ordinal |
| Oblique random forest | 0.667 ± 0.011 (+0.027) | 0.682 ± 0.015 (+0.009) | 0.680 | n_components=6, tree max_depth=8, tree min_samples_leaf=86, impute=mean, encoder=target |
| AdaBoost | 0.676 ± 0.012 (+0.006) | 0.681 ± 0.016 (+0.006) | 0.683 | n_estimators=153, learning_rate=0.020, tree max_depth=3, impute=median, encoder=target |

On nested CV, the oblique random forest (0.682 ± 0.015) and AdaBoost (0.681 ± 0.016) have the highest means, followed by the random forest (0.676), the decision tree (0.674) and logistic regression (0.671), but the whole range is 0.011, smaller than the fold-to-fold std of 0.010-0.016, and the corrected paired test finds no significant difference between any two models, so I would choose the oblique forest only as a tie-break with AdaBoost. The chosen hyperparameters fit how each model works: AdaBoost used depth-3 trees with a very small learning rate (0.02) over 153 rounds, many small steps of weak learners; the decision tree needs a large leaf (94 rows) to close the 0.085 train-validation gap of the unconstrained tree; and the oblique forest went from one projection component to six, which lifted it from 0.667 to 0.682, and it settled on moderately deep trees with large leaves (86 rows). The nested folds disagree widely on the oblique forest’s depth (3 to 29) and leaf size (1 to 61), so those values are poorly determined, because the signal is weak and every model tops out near 0.68. I would report the oblique forest’s nested CV score, 0.682 ± 0.015, not a tuning score, because tuning scores are the best of many noisy trials and are optimistic. The locked test set stays untouched until the final assessment.

### Ensemble (with feature engineering)

| Model | Default: CV mean ± std (gap) | Tuned: nested CV mean ± std (gap) | Tuning score (best trial) | Chosen hyperparameters |
|---|---|---|---|---|
| Logistic regression | 0.678 ± 0.013 (+0.001) | 0.674 ± 0.010 (+0.006) | 0.680 | C=1.720, class_weight=None, impute=mean, scaler=none, encoder=target |
| Decision tree | 0.600 ± 0.012 (+0.108) | 0.667 ± 0.013 (+0.017) | 0.673 | max_depth=4, min_samples_leaf=3, criterion=entropy, impute=median, encoder=target |
| Random forest | 0.646 ± 0.011 (+0.102) | 0.674 ± 0.008 (+0.016) | 0.681 | max_depth=9, min_samples_leaf=17, max_features=sqrt, class_weight=None, impute=median, encoder=onehot |
| Oblique random forest | 0.666 ± 0.009 (+0.052) | 0.675 ± 0.013 (+0.021) | 0.680 | n_components=5, tree max_depth=7, tree min_samples_leaf=14, impute=median, encoder=target |
| AdaBoost | 0.675 ± 0.009 (+0.006) | 0.676 ± 0.014 (+0.007) | 0.681 | n_estimators=103, learning_rate=0.419, tree max_depth=3, impute=mean, encoder=ordinal |

The introduction of engineered features (priors_per_adult_year, log_priors, and young_x_priors) did not raise the performance ceiling; in fact, the nested cross-validation scores plateaued slightly lower than the baseline models without feature engineering. AdaBoost and the Oblique Random Forest remain the top performers among the ensembles, reaching 0.676 and 0.675 on nested CV, respectively. However, this represents a slight drop compared to their previous scores of 0.681 and 0.682.   

The new interaction and ratio features failed to expose hidden signal that the models weren't already capturing from the raw age and priors_count columns. Instead, the added complexity slightly narrowed the train-validation gap for the default Logistic Regression (from +0.003 to +0.001) but widened the gap for the tuned decision tree (from +0.011 to +0.017).


