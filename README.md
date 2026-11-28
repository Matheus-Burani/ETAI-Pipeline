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

This table is updated after each practical class, so you can always see what changed in the pipeline and why -- it's a running log, not a fixed syllabus.

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | EDA + preprocessing -- diagnose the data, then fix it | `src/data_diagnostics.py` (missingness-mechanism test via chi-square + Cramér's V, domain-rule invalid-value detection, two-way duplicate check) and `src/preprocessing.py` (leak-safe category cleanup, mechanism-matched imputation with `_was_missing` indicators for MNAR columns, a deployable `ColumnTransformer`, **and** the train/test split itself, all in the one file rather than split across two) replace the old naive `dropna()`/`pd.get_dummies()` preprocessing; encoder/scaler pair (target encoding + standard scaling) chosen by an empirical grid over 15 repeated splits, checked against the runner-up with a paired comparison so the win isn't just noise; three redundant columns (found via correlation + VIF) dropped; `config.yaml` gains `diagnostics` and `preprocessing` sections -- see "Preprocessing decisions" below. Threshold-independent metrics (ROC-AUC/PR-AUC) and a calibration check are deliberately **not** added yet -- not yet |
| 4 | Preprocessing inside the pipeline + cross-validation -- evaluating a model honestly | A **locked final test set** (20%, stratified, seed 42) is set aside by `split_dev_test()` (replaces `split_train_test()`) and never scored; models are now judged by **stratified 5-fold cross-validation** of the whole pipeline (preprocessing + model) on the development set, reported per fold with mean ± std and the train-validation gap; the classification report and fairness check now use out-of-fold predictions; target encoding switched to scikit-learn's cross-fitting `TargetEncoder` (a row's own label never leaks into its own encoding), encoder/scaler set by hand in `config.yaml` (target encoding + robust scaling, reasons in the comments); **two fixes** in `clean_dataset()`: genuine `NaN`s in categorical columns were being turned into the string `"nan"` (a fake category), so 229 `c_charge_degree` gaps were never imputed or flagged -- fixed in `config.yaml` alone: `"nan"` added to `diagnostics.placeholder_tokens` (the category cleanup's last step turns listed tokens into `NaN`, after its text conversion); and it no longer drops rows -- de-duplication moved to a separate, training-only `drop_duplicate_rows()` (run before the dev/test split), so the same cleaning can run on new data where every row needs a prediction; `src/data_diagnostics.py` removed -- its one cleaning function (`flag_invalid_values`) moved into `preprocessing.py`, and the EDA-only checks (missingness test, duplicate counts) live in the EDA notebooks, not in every pipeline run; `dummy` (majority-class) model added as the floor to beat, and `random_forest` registered (sensible defaults, untuned); the final model is refit on the whole development set after CV; `config.yaml` gains `test_set` and `cv` sections -- see "Model evaluation" below |
| 5 | Hyperparameter tuning without leakage | New `src/tuning.py`: hyperparameters are tuned with **Optuna** (seeded TPE sampler) on the development set only -- every candidate is scored by stratified 5-fold CV of the **whole** pipeline (preprocessing re-fit in every fold of every trial), on the same folds for every candidate; the tuning procedure is estimated honestly by **nested cross-validation** (the tuning repeated inside each outer CV fold, scored on the outer fold it never saw), whose out-of-fold predictions now feed the classification report and fairness check; the reported "best trial" score is shown next to the nested one as an optimism check; the final model is the tuned pipeline refit on all development rows; `build_pipeline()` in `src/model.py` now builds the preprocessing + model `Pipeline` (one place to add a learned step later); `config.yaml` gains a `tuning` section (on/off, trials, inner folds, seed, one search space per model type) and the model becomes the decision tree; `optuna` added to `requirements.txt`. The locked test set is still untouched -- see "Hyperparameter tuning" below |
| 6 | Ensembles in depth: random forest, oblique random forest, AdaBoost | Two new model types in `src/model.py`: `oblique_random_forest` (a forest of oblique trees built from scikit-learn parts -- every tree splits on its own random sparse linear combinations of the features, via `SparseRandomProjection` inside `BaggingClassifier`) and `adaboost` (AdaBoost over decision trees of a chosen depth; stumps by default), both registered in `_MODELS`; the random forest is now the model `main.py` runs (200 trees, all CPU cores); `config.yaml` gains search spaces for the three ensembles -- the forests tune tree size (`max_depth`, `min_samples_leaf`) and decorrelation (`max_features` / number of combinations) but not the number of trees, AdaBoost tunes its base trees' depth, its number of rounds and its learning rate; `corrected_paired_test()` in `src/evaluate.py` compares two models scored on the same folds (Nadeau & Bengio's corrected paired t-test), so a "winner" is only called when the difference is bigger than fold-to-fold noise; `scipy` listed in `requirements.txt` (it already came with scikit-learn, so nothing new to install) -- see "Ensembles" below |
| 7 | Gradient boosting + evaluating with probabilities | Three new model types in `src/model.py`: `gradient_boosting` (scikit-learn's `GradientBoostingClassifier`: each shallow tree fits the gradient of the log-loss left by the trees before it), `hist_gradient_boosting` (the same idea on binned features: much faster on large data, and it handles missing values itself) and `naive_bayes` (`GaussianNB`, a fast probabilistic baseline); gradient boosting is now the model `main.py` runs, each with its search space in `config.yaml`; evaluation now reports **several metrics** (accuracy, balanced accuracy, precision, recall, F1, F-beta, ROC-AUC, PR-AUC) and one **selection metric** (`cv.selection_metric`, ROC-AUC) that tuning maximises and runs are ranked by; out-of-fold **probabilities** (not just labels) feed the reports, with a confusion matrix; `class_weight: balanced` in the commented examples of the models that support it; new `save_scores()` in `src/results.py`: every run adds a row to `results/runs.csv` (the model and its settings, the score on each fold, the no-model floor and the untuned score on the same folds), so any two runs on the same data and folds can be compared with `corrected_paired_test()`; the printed output was reorganised to be easier to read, with the full tables in the results file (optional: keep your own format if you prefer) -- see "Model evaluation" below |
| 8 | XGBoost + feature selection | New model type `xgboost` (`XGBClassifier`, from the `xgboost` package: gradient boosting with extra regularisation, multi-threaded, handles missing values itself), now the model `main.py` runs, with its search space in `config.yaml`; **feature selection** as a learned step of the Pipeline: new `VoteSelector` class in `src/model.py` (a random forest, gradient boosting and an L1 logistic regression vote; features with at least `min_votes` votes are kept), placed between `"prep"` and `"model"` by `build_pipeline()` when the new `feature_selection` section of `config.yaml` switches it on, so it is re-fitted on the training rows of every fold (never on the rows it is scored on); with tuning on, `select__min_votes` is tuned with the model and assessed by nested CV; the pipeline caches fitted preprocessing and selection steps during a run (`Pipeline(memory=...)`, a temporary folder deleted at the end), so tuning doesn't refit them in every trial; the final model's kept features are printed, and the scoreboard records the selection setting -- see "Feature selection" below |
| 9 | Beyond COMPAS -- Option B: three-class target (`score_text`) | Generalized the binary-only pipeline to support multiclass classification (`Low`, `Medium`, `High`) via `config_multiclass.yaml` while keeping `python main.py` backwards-compatible for binary COMPAS (`sys.argv[1]` selects the config file); text labels mapped to ordered integers `0, 1, 2` via `data.class_names` in `split_features_target()` (required by `XGBClassifier` and class-ordered reports); `TargetEncoder` switched from `target_type="binary"` to `"auto"`; `VoteSelector` updated to check L1 coefficients across all class rows (`.any(axis=0)`) instead of only `coef_[0]`; `report_proba()` added so `cross_validate_pipeline()` and `nested_cross_validate()` store a 1D probability array for 2 classes or a 2D `rows x classes` matrix for 3+ classes; `build_scorers()` extended with `fbeta_macro_<beta>` and multiclass metrics (`roc_auc_ovr`, `f1_macro`); `oof_classification_report()` and `fairness_report()` updated to use `argmax` predictions, a 3x3 labeled confusion matrix, and a false-High-rate demographic check when `oof_proba.ndim == 2` -- see "Beyond COMPAS" below |
| 10 | Missing values + a deployable pipeline | The numeric imputer is a config choice (`preprocessing.imputation.numeric_strategy`): `median`, `mean`, `knn` (`KNNImputer`), `iterative` (MICE with one completed dataset, `IterativeImputer`) or `random_forest` (the missForest idea: `IterativeImputer` with a random forest per column); the numeric columns are now scaled **before** they are imputed (scalers ignore gaps when fitting, and KNN needs one scale); none of the imputers looks at the target, so they also run on new data; the runs table and report record the imputer; `main.py` **saves the final fitted pipeline** to `models/model.joblib`; new **`predict.py`**: `python predict.py new.csv predictions.csv` cleans the file with the same rules (no rows dropped) and writes one row per input row (id, probability, prediction); `data/compas_new_people.csv` to try it -- see "Missing values" and "Predicting new data" below |
| 11 | Outliers: Isolation Forest and LOF | New `AnomalyScore` step in `src/preprocessing.py`: an Isolation Forest (`-score_samples`, the theory's s(x)) or a Local Outlier Factor (`novelty=True`, so new people can be scored; training rows get their proper `negative_outlier_factor_`) fitted on the training rows of each fold, appending one column, `anomaly_score`, after the preprocessor and before feature selection; `build_pipeline()` takes an `anomaly_config`, switched on by the new `anomaly` section of `config.yaml` (`method: none` by default); the runs table, scoreboard and report record the anomaly method; `predict.py` is unchanged (the saved Pipeline contains the fitted detector); new `check_anomalies.py` (reoffend rate of the most unusual people); `compare_imputers.py --by anomaly` compares runs by anomaly method -- see "Outliers" below |
| 11 | Outliers and anomaly detection | New `AnomalyScore` class in `src/preprocessing.py`: an Isolation Forest or a Local Outlier Factor (`novelty=True`, so it can score new rows) fitted on the training rows of each fold, which appends an `anomaly_score` column (higher = more unusual) to the preprocessed features; `build_pipeline()` places it right after `"prep"` (before feature selection, which can keep or drop it) when the new `anomaly` section of `config.yaml` sets `method` to `isolation_forest` or `lof` (`none` by default); the runs table, the printed header and the report record the anomaly method; nothing changes in `predict.py`: the saved Pipeline already contains the fitted detector -- see "Anomaly scores" below |


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

With the virtual environment active, from the project root:

```bash
# Run the original binary COMPAS pipeline (defaults to config.yaml)
python main.py

# Run Option B: three-class COMPAS risk category prediction
python main.py config_multiclass.yaml
```


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


## Week 7

# Model evaluation

**Metrics.** `cv.metrics` lists what every run reports; `cv.selection_metric` is the one that matters: tuning maximises it and the scoreboard ranks runs by it. The default is **ROC-AUC**: it scores how well the model *ranks* people by risk, without choosing a threshold, so COMPAS's own 1-10 risk score can be judged the same way. Accuracy, precision, recall and F1 depend on the threshold that turns a probability into a yes/no (0.5 in this pipeline). `fbeta_<beta>` is F-beta: `fbeta_2` counts recall twice as much as precision.

**The scoreboard.** Every `python main.py` adds one row to `results/runs.csv` (`save_scores()` in `src/results.py`): the run id, the data file, the model, whether it was tuned, the encoder and scaler, the model's params (and the tuned ones), the selection metric with its mean and std, the **untuned** score of the same model, the **no-model floor** (a dummy on the same folds) and the score on each fold (`fold_1` ... `fold_5`, from nested CV when tuning is on). The console prints the runs on the same data and metric, best first. The full report of each run is in `results/run_<run id>.txt`.

Two runs can only be compared if they used the **same data, cleaning and folds**: when you change the data file, the cleaning rules or `cv.random_state`, delete `results/runs.csv` and start a new table. (You could make this more automatic.) To compare two runs, from the project root:

```python
import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
corrected_paired_test(runs.loc[0, folds], runs.loc[1, folds], test_train_ratio=1 / 4)   # rows 0 and 1; 5-fold: 1 / (5 - 1)
```

It returns the mean difference, the folds the first run wins, t and the p-value.


# Results week 7

## Week 7 – Model comparison with ROC-AUC as the selection metric

### What changed this week
- The selection metric changed from accuracy to **ROC-AUC**, which measures how well a model ranks people by risk independently of any threshold. The no-model floor is therefore 0.500 (the earlier accuracy floor of 0.549 no longer applies).
- Eight models were evaluated with the same 5-fold stratified CV, 30 Optuna trials and nested CV: naive Bayes, logistic regression, decision tree, random forest, oblique random forest, AdaBoost, gradient boosting and hist gradient boosting.
- The tuning search now also covers preprocessing (imputation strategy, categorical encoding and, for logistic regression, scaling).
- The locked test set (1,443 rows) is still unevaluated.

### Results (ROC-AUC, development set, 5,771 rows)

| Model | Default CV | Tuned nested CV (mean ± std) | Tuning gain | Optimism | Default gap | Tuned gap |
|---|---|---|---|---|---|---|
| Gradient boosting | 0.727 | **0.727 ± 0.014** | 0.000 | ≈0.000 | +0.022 | +0.012 |
| Hist gradient boosting | 0.717 | **0.727 ± 0.016** | +0.010 | ≈0.000 | +0.058 | +0.014 |
| AdaBoost | 0.723 | 0.726 ± 0.014 | +0.003 | -0.001 | +0.005 | +0.014 |
| Oblique random forest | 0.685 | 0.726 ± 0.016 | +0.041 | ≈0.000 | +0.150 | +0.020 |
| Random forest | 0.683 | 0.726 ± 0.018 | +0.043 | ≈0.000 | +0.121 | +0.024 |
| Logistic regression | 0.719 | 0.722 ± 0.016 | +0.003 | ≈0.000 | +0.002 | +0.001 |
| Decision tree | 0.604 | 0.719 ± 0.014 | +0.115 | -0.001 | +0.110 | +0.016 |
| Naive Bayes | 0.685 | 0.705 ± 0.012 | +0.020 | ≈0.000 | +0.002 | +0.001 |
| Dummy (floor) | 0.500 | – | – | – | – | – |

Chosen hyperparameters (tuning on the whole development set):

| Model | Hyperparameters |
|---|---|
| Gradient boosting | n_estimators=426, learning_rate=0.012, max_depth=3, min_samples_leaf=2, subsample=0.77 |
| Hist gradient boosting | max_iter=82, learning_rate=0.111, max_leaf_nodes=4, min_samples_leaf=130, l2=0.0007 |
| AdaBoost | base tree max_depth=5, n_estimators=254, learning_rate=0.052, mean imputation, ordinal encoding |
| Oblique random forest | max_depth=20, min_samples_leaf=127, n_components=5, median imputation, target encoding |
| Random forest | max_depth=15, min_samples_leaf=47, max_features=0.5, median imputation, ordinal encoding |
| Logistic regression | C=0.0008, class_weight=None, mean imputation, robust scaling, ordinal encoding |
| Decision tree | max_depth=6, min_samples_leaf=81, gini, median imputation, one-hot encoding |
| Naive Bayes | var_smoothing=0.091 |

**Interpretation.** All models are well above the 0.500 floor (about +0.22 for the top group), so they learn real signal. The six best models are separated by only 0.008 ROC-AUC (0.719 to 0.727), less than the fold-level standard deviation (0.014 to 0.018), so they are statistically tied. Gradient boosting is nominally first, but its edge over logistic regression (+0.005) is small, even though it was ahead in four of five outer folds and tied in the other. Naive Bayes is the only model clearly behind, about 0.02 below the rest, which is consistent with its independence assumption not holding for correlated features.

Tuning matters most for the models that overfit by default. The decision tree gains +0.115 and the random forests about +0.04, because tuning shrinks their train/validation gaps from +0.11 to +0.15 down to +0.016 to +0.024. Logistic regression, AdaBoost and gradient boosting gain almost nothing because their defaults were already well regularized. The best configurations are all simple: shallow trees (depth 3 for gradient boosting, 4 leaves for hist gradient boosting) with low learning rates, and logistic regression with very strong regularization (C≈0.0008). The optimism check is ≈0 for every model, so tuning is not inflating the scores, and the top five Optuna trials of each model score within 0.001 to 0.002 of each other, so the tuning surface is flat. The chosen preprocessing is also unstable (the decision tree chose one-hot encoding on the full data, but ordinal or target encoding in the nested folds), so I would not read much into those choices.

All models share the same hard fold (fold 2, ROC-AUC 0.69 to 0.70), which points to the data split rather than the model.

### Classification metrics and fairness (threshold 0.5, out-of-fold, class 1 = reoffended)

| Model | Accuracy | Precision | Recall | F1 | FPR African-American | FPR Caucasian | Ratio | Difference |
|---|---|---|---|---|---|---|---|---|
| Gradient boosting | 0.68 | 0.68 | 0.56 | 0.61 | 0.29 | 0.16 | 1.81x | 0.13 |
| Hist gradient boosting | 0.67 | 0.64 | 0.64 | 0.64 | 0.38 | 0.23 | 1.65x | 0.15 |
| AdaBoost | 0.68 | 0.67 | 0.56 | 0.61 | 0.30 | 0.16 | 1.88x | 0.14 |
| Oblique random forest | 0.68 | 0.67 | 0.56 | 0.61 | 0.30 | 0.16 | 1.88x | 0.14 |
| Random forest | 0.67 | 0.64 | 0.63 | 0.64 | 0.37 | 0.22 | 1.68x | 0.15 |
| Logistic regression | 0.65 | 0.71 | 0.39 | 0.50 | 0.19 | 0.09 | 2.11x | 0.10 |
| Decision tree | 0.67 | 0.64 | 0.64 | 0.64 | 0.38 | 0.24 | 1.58x | 0.14 |
| Naive Bayes | 0.62 | 0.74 | 0.24 | 0.37 | 0.11 | 0.04 | 2.75x | 0.07 |
| COMPAS | – | – | – | – | 0.45 | 0.23 | 1.96x | 0.22 |

**Interpretation.** The models split into two groups that reflect the operating point, not the algorithm. The decision tree, random forest and hist gradient boosting keep `class_weight='balanced'` from the config, so they flag more people (recall 0.63 to 0.64, precision 0.64, overall FPR about 0.30). Gradient boosting, AdaBoost and the oblique random forest have no class weighting and are more conservative (recall 0.56, precision 0.67 to 0.68, overall FPR about 0.22). Logistic regression is a special case: its search space included `class_weight`, ROC-AUC does not reward it, and tuning dropped it in four of five folds. That is why its recall fell to 0.39 (and varies widely across folds, std 0.115) even though its ROC-AUC barely changed. Naive Bayes is the extreme case, flagging almost nobody (recall 0.24).

On fairness, every model still flags non-reoffending African-American defendants more often than Caucasian ones. The absolute gap (0.07 to 0.15) is smaller than COMPAS's (0.22), but ratios of 1.6x to 2.8x show the disparity is not removed. The ratio alone is misleading: the highest ratios (naive Bayes 2.75x, logistic regression 2.11x) belong to the models that flag the fewest people, and have the smallest absolute gaps, while the lowest ratios (decision tree 1.58x, hist gradient boosting 1.65x) belong to the models that flag the most. The fairness differences between models therefore mostly follow the threshold, and they should be compared at a common operating point. The Asian (n=21), Native American (n=6) and unknown (n=66) groups are too small to interpret.

### Conclusions
1. The signal is limited. Eight different model families plateau at about 0.73 ROC-AUC and 0.68 accuracy, consistent with the earlier weeks, so the features rather than the algorithm are the bottleneck.
2. There is no statistically meaningful winner among the top six. Gradient boosting is nominally best, but logistic regression is within 0.005 and is far simpler and easier to explain.
3. Model choice does not fix the racial FPR disparity. It mainly changes how many people get flagged, and the threshold is what trades recall against false positives.


# Week 8

## Feature selection

Set feature_selection.enabled in config.yaml to add the "select" step between preprocessing and the model. VoteSelector fits three models on the training rows (a random forest, gradient boosting, and an L1-penalised logistic regression); each votes for the features it finds useful (above-average importance, or a non-zero coefficient), and features with at least min_votes votes are kept. Because it learns from the labels, it lives inside the Pipeline and is re-fitted in every fold; selecting features on all rows before cross-validation would make the score optimistic. With tuning on, feature_selection.search_space is added to the model's search space, so min_votes is chosen with the model's hyperparameters and assessed by nested CV. The features the final model keeps are printed at the end of each run.

XGBoost is a separate package: see "Installing XGBoost" below if import xgboost fails.

### Installing XGBoost
xgboost is in requirements.txt, so pip install -r requirements.txt (venv active) installs it. On its own: pip install xgboost. Check it worked, with the venv active:

python -c "import xgboost; print(xgboost.__version__)"
Windows: nothing extra: pip install xgboost installs a ready-made version. If it fails to build or install, update pip first (python -m pip install --upgrade pip) and make sure your Python is 64-bit (python -c "import struct; print(struct.calcsize('P') * 8)" should print 64).

macOS: XGBoost needs the OpenMP library (for running on several CPU cores), which macOS doesn't ship. If import xgboost fails with XGBoostError, Library not loaded: libomp.dylib or similar:

install Homebrew if you don't have it (the command is on its home page);
run brew install libomp;
open a new terminal, activate the venv, and try the import again.
Works on both Apple Silicon (M1/M2/M3/M4) and Intel Macs. If you use conda instead of a venv, conda install -c conda-forge xgboost installs XGBoost together with its OpenMP library.

Linux: on most distributions pip install xgboost just works. Two things can go wrong:

it downloads a very large extra package (nvidia-nccl-..., a few hundred MB) for NVIDIA GPUs. If you have no NVIDIA GPU, or the download is too slow, install the CPU-only version instead: pip install xgboost-cpu (the code is identical: still import xgboost). If you do this, replace xgboost with xgboost-cpu in your copy of requirements.txt, so a reinstall doesn't fetch the large package again.
a very old distribution may have no ready-made version and try to compile it. Update pip first (python -m pip install --upgrade pip); recent releases need glibc 2.28 or newer (Ubuntu 20.04+, Debian 10+, Fedora 29+).
In a notebook (Jupyter / VS Code): No module named 'xgboost', although you installed it. The notebook runs on a different Python from the one pip installed into (e.g. you installed into the venv, but the notebook uses the system Python). Either:

install from inside the notebook, into the Python it actually uses: run %pip install xgboost in a cell, then restart the kernel; or
make your venv available as a kernel, once, with the venv active:
pip install ipykernel
python -m ipykernel install --user --name etai --display-name "ETAI (venv)"
then choose ETAI (venv) as the notebook's kernel. The notebook then sees every package in requirements.txt.
To check which Python a notebook is using: import sys; print(sys.executable). In the terminal: which python (macOS/Linux) or where python (Windows). They should point into the same venv folder.


### Features kept
Every final model keeps the same **2 of 10 features: `age` and `priors_count`**, regardless of whether the tuned `min_votes` was 1, 2 or 3. The logs list only the kept features, not the dropped ones.

### Results (ROC-AUC, development set, 5,771 rows)

| Model | Week 7 tuned (all features) | Default CV | Tuned nested CV (mean ± std) | Change vs Week 7 | Tuning gain | Optimism | Default gap | Tuned gap |
|---|---|---|---|---|---|---|---|---|
| Gradient boosting | 0.727 | 0.723 | **0.726 ± 0.013** | -0.001 | +0.003 | -0.001 | +0.019 | +0.005 |
| XGBoost | – | 0.721 | 0.725 ± 0.012 | new | +0.004 | ≈0.000 | +0.027 | +0.006 |
| Hist gradient boosting | 0.727 | 0.710 | 0.725 ± 0.013 | -0.002 | +0.015 | ≈0.000 | +0.051 | +0.008 |
| Oblique random forest | 0.726 | 0.672 | 0.724 ± 0.012 | -0.002 | +0.052 | -0.001 | +0.106 | +0.014 |
| Random forest | 0.726 | 0.672 | 0.724 ± 0.013 | -0.002 | +0.052 | -0.002 | +0.102 | +0.014 |
| AdaBoost | 0.726 | 0.721 | 0.723 ± 0.013 | -0.003 | +0.002 | +0.001 | +0.005 | +0.012 |
| Decision tree | 0.719 | 0.657 | 0.718 ± 0.014 | -0.001 | +0.061 | +0.001 | +0.117 | +0.016 |
| Logistic regression | 0.722 | 0.716 | 0.717 ± 0.011 | -0.005 | +0.001 | +0.001 | +0.000 | +0.001 |
| Naive Bayes | 0.705 | 0.704 | 0.708 ± 0.013 | +0.003 | +0.004 | ≈0.000 | +0.001 | +0.001 |
| Dummy (floor) | 0.500 | 0.500 | – | – | – | – | – | – |

Chosen hyperparameters (tuning on the whole development set; all models use `age` and `priors_count`):

| Model | Hyperparameters |
|---|---|
| Gradient boosting | n_estimators=171, learning_rate=0.015, max_depth=3, min_samples_leaf=19, subsample=0.53, min_votes=3 |
| XGBoost | n_estimators=156, learning_rate=0.177, max_depth=5, min_child_weight=2.28, subsample=0.67, colsample_bytree=0.62, reg_lambda=0.20, reg_alpha=0.34, gamma=3.52, min_votes=1 |
| Hist gradient boosting | max_iter=82, learning_rate=0.113, max_leaf_nodes=4, min_samples_leaf=7, l2=0.0004, min_votes=1 |
| Oblique random forest | max_depth=23, min_samples_leaf=169, n_components=10, min_votes=2 |
| Random forest | max_depth=23, min_samples_leaf=74, max_features=1.0, min_votes=2 |
| AdaBoost | base tree max_depth=3, n_estimators=398, learning_rate=0.100, min_votes=2 |
| Decision tree | max_depth=5, min_samples_leaf=129, gini, min_votes=3 |
| Logistic regression | C=0.0018, class_weight=balanced, mean imputation, robust scaling, target encoding, min_votes=3 |
| Naive Bayes | var_smoothing=0.099, min_votes=2 |

**Interpretation.** Removing eight of the ten features changed the tuned ROC-AUC by at most 0.005 for any model, well inside the fold-level standard deviation (0.011 to 0.014). Two features therefore reproduce the ranking power of ten, and the other eight add essentially nothing. Naive Bayes is the only model that improved (+0.003), which fits its independence assumption, since fewer correlated inputs help it. Logistic regression has the largest drop (-0.005), still within noise.

The top eight models span only 0.009 ROC-AUC (0.717 to 0.726), less than one standard deviation, so they are statistically tied. Gradient boosting is nominally first, but boosted trees on two features perform the same as a heavily regularized logistic regression on two features, which suggests there is little interaction structure to exploit. Naive Bayes remains clearly behind (about 0.015 to 0.018 below the rest).

Untuned scores moved more than tuned ones. The decision tree improved from 0.604 to 0.657 because fewer features means less to overfit, while the untuned forests dropped (random forest 0.683 to 0.672, oblique random forest 0.685 to 0.672), probably because two features give the trees little diversity to exploit. After tuning these differences disappear. Tuning still matters most for the models that overfit by default (decision tree +0.061, forests +0.052), because it shrinks their train/validation gap from about +0.10 to +0.12 down to +0.014 to +0.016. The optimism check is between -0.002 and +0.001 for every model, so tuning is not inflating the scores. All models again share the same hardest fold (fold 2, ROC-AUC 0.688 to 0.705), which points to the data split rather than the model.

### Classification metrics and fairness (threshold 0.5, out-of-fold, class 1 = reoffended)

| Model | Accuracy | Precision | Recall | F1 | FPR African-American | FPR Caucasian | Ratio | Difference |
|---|---|---|---|---|---|---|---|---|
| Gradient boosting | 0.68 | 0.68 | 0.56 | 0.61 | 0.29 | 0.16 | 1.81x | 0.13 |
| XGBoost | 0.68 | 0.67 | 0.57 | 0.62 | 0.30 | 0.16 | 1.88x | 0.14 |
| Hist gradient boosting | 0.67 | 0.63 | 0.66 | 0.65 | 0.41 | 0.25 | 1.64x | 0.16 |
| Oblique random forest | 0.68 | 0.67 | 0.56 | 0.61 | 0.30 | 0.16 | 1.88x | 0.14 |
| Random forest | 0.67 | 0.64 | 0.62 | 0.63 | 0.37 | 0.22 | 1.68x | 0.15 |
| AdaBoost | 0.68 | 0.67 | 0.56 | 0.61 | 0.30 | 0.16 | 1.88x | 0.14 |
| Decision tree | 0.67 | 0.65 | 0.59 | 0.62 | 0.34 | 0.20 | 1.70x | 0.14 |
| Logistic regression | 0.66 | 0.66 | 0.53 | 0.59 | 0.30 | 0.17 | 1.76x | 0.13 |
| Naive Bayes | 0.63 | 0.73 | 0.28 | 0.41 | 0.13 | 0.05 | 2.60x | 0.08 |
| COMPAS | – | – | – | – | 0.45 | 0.23 | 1.96x | 0.22 |

**Interpretation.** As in Week 7, the differences between models mostly reflect the operating point rather than the algorithm. Configurations that keep `class_weight='balanced'` (decision tree, random forest, hist gradient boosting) flag more people (recall 0.59 to 0.66, FPR for African-American defendants 0.34 to 0.41), and the unweighted models (gradient boosting, XGBoost, AdaBoost, oblique random forest) are more conservative (recall 0.56 to 0.57, FPR about 0.30). Naive Bayes flags the fewest people (recall 0.28). Logistic regression moved from recall 0.39 in Week 7 to 0.53, because most outer folds now chose `class_weight='balanced'`. Its recall is still unstable across folds (std 0.133), since fold 1 chose no class weighting (recall 0.30) and folds 2 to 5 chose balanced (0.58 to 0.63). The pooled figure therefore mixes two settings and is not a stable property of the model.

On fairness, every model still flags non-reoffending African-American defendants more often than Caucasian ones (ratios 1.6x to 2.6x). The absolute gap (0.08 to 0.16) is smaller than COMPAS's (0.22), but the disparity is not removed. The highest ratio belongs to the model that flags the fewest people (naive Bayes), so models should be compared at a common threshold. The disparity persists even though only `age` and `priors_count` are used. A plausible explanation is that these two variables are correlated with race in this data, but the logs do not test this. The Asian (n=21), Native American (n=6) and unknown (n=66) groups are too small to interpret.

### Conclusions
1. `age` and `priors_count` carry nearly all of the usable signal. Eight of ten features can be dropped with no meaningful loss (at most 0.005 ROC-AUC).
2. The top eight models are statistically tied at about 0.72 to 0.73 ROC-AUC, so the choice among them should rest on simplicity and operating point. A two-feature logistic regression or a shallow tree is easy to justify and to explain.
3. Feature selection does not change the racial FPR disparity, and model choice mainly changes how many people are flagged.
# Week 9

## Beyond COMPAS (Option B: Three-Class Target)

For Week 9, the pipeline was extended to solve **Option B**: predicting COMPAS's own three-class risk category (`score_text`: **Low** 54.0%, **Medium** 26.5%, **High** 19.4%) from the same demographic and criminal-history features, while keeping the original binary COMPAS pipeline (`two_year_recid`) working unchanged.

### How to run the configs

With the virtual environment active, from the project root:

```bash
# Run the original binary COMPAS pipeline (defaults to config.yaml)
python main.py

# Run Option B: three-class COMPAS risk category prediction
python main.py config_multiclass.yaml
```

Multiclass runs are saved to their own output directory (`results_multiclass/` and `results_multiclass/runs.csv`) so the scoreboard does not mix binary and three-class runs.

### What broke, why it broke, and how it was fixed

| Component / File | What broke | Why it broke (hidden binary assumption) | How it was fixed |
|---|---|---|---|
| **Entry point** (`main.py`) | Could only run `config.yaml` and did not pass class names to reports | `load_config()` was called with no arguments; report functions lacked `class_names` | Read `sys.argv[1]` (defaulting to `"config.yaml"`) and passed `config["data"].get("class_names")` to `oof_classification_report()` and `fairness_report()` |
| **Feature & target split** (`config_multiclass.yaml`, `src/preprocessing.py`) | Target leakage risk, trailing space in `"score_text "`, and string labels (`"Low"`, `"Medium"`, `"High"`) crashed `XGBClassifier` and broke class ordering | `decile_score` (the 1–10 number behind `score_text`) gives the answer away, and `two_year_recid` is a future outcome not known when the score was given; alphabetical sorting turns `"High", "Low", "Medium"` into `0, 1, 2` | Dropped `decile_score` and `two_year_recid` in `config_multiclass.yaml`; mapped `y` to `0, 1, 2` in exact risk order (`["Low", "Medium", "High"]`) using `data.class_names` in `split_features_target()` |
| **Target encoder** (`src/preprocessing.py`) | `ValueError` during preprocessing when fitting `TargetEncoder` on 3 classes | `_ENCODERS["target"]` hardcoded `TargetEncoder(target_type="binary")` | Changed `target_type="binary"` to `target_type="auto"` so scikit-learn uses binary encoding for 2 classes and multiclass encoding (one column per class) for 3+ classes |
| **Feature selection** (`src/model.py`) | `ValueError` from `solver="liblinear"` on 3 classes, and `VoteSelector` ignored features used by classes 1 and 2 | `liblinear` does not support `n_classes >= 3` in modern scikit-learn; in binary classification `lasso.coef_` has 1 row (`coef_[0]`), whereas with 3 classes it has shape `(3, n_features)` | Switched L1 `LogisticRegression` to `solver="saga", max_iter=2000` and replaced `np.abs(lasso.coef_[0]) > 1e-8` with `(np.abs(lasso.coef_) > 1e-8).any(axis=0)` |
| **CV & nested CV probabilities** (`src/evaluate.py`, `src/tuning.py`) | `ValueError` / lost class probabilities when storing `oof_proba` | Both `cross_validate_pipeline()` and `nested_cross_validate()` initialized `oof_proba = np.empty(len(X))` (1D) and sliced `predict_proba(X)[:, 1]` | Used `report_proba()` to return a 1D array for 2 classes or the full `(n_rows, n_classes)` matrix for 3+ classes, and initialized `oof_proba = np.empty((len(X),) + proba.shape[1:])` |
| **Metrics & scorers** (`config_multiclass.yaml`, `src/evaluate.py`) | Binary metrics (`f1`, `precision`, `recall`, `roc_auc`, `fbeta_2`) failed on 3 classes | Binary metrics require a single positive class and a 1D probability vector | Switched config metrics to `roc_auc_ovr`, `f1_macro`, `precision_macro`, `recall_macro`, `accuracy`, `balanced_accuracy`, and added `fbeta_macro_<beta>` support in `build_scorers()` |
| **Classification & fairness reports** (`src/evaluate.py`) | Thresholding (`oof_proba >= 0.5`), 2x2 confusion matrix, and binary FPR comparison against `score_text` failed | With 3 classes there is no single threshold on `P(1)`, the confusion matrix is 3x3, and `score_text` is now the target itself | When `oof_proba.ndim == 2`, predictions use `argmax(axis=1)`, the confusion matrix is 3x3 labeled with `class_names`, and `_multiclass_fairness_report()` computes the false-`High` rate per racial group |

### Feature selection and class weighting decisions

**When `feature_selection` helps vs. hurts:** With `target_type="auto"`, target encoding expands the 3 categorical features into 9 columns (16 total features), while ordinal encoding produces 10 total features. Comparing `feature_selection: off` against `VoteSelector` (`min_votes=1`) showed that keeping all features works better for regularized linear models and tree ensembles (`logistic_regression` rose from `0.759` to `0.761` `roc_auc_ovr` and `0.534` to `0.541` `f1_macro`, while `xgboost`, `gradient_boosting`, `hist_gradient_boosting`, and `random_forest` gained `+0.005` to `+0.013` `roc_auc_ovr` by retaining all features for subsampling and splits). Conversely, `VoteSelector` (`min_votes=1`, keeping **6 of 16 features**: `numeric__age`, `numeric__juv_fel_count`, `numeric__juv_misd_count`, `numeric__priors_count`, `categorical__age_cat_0`, `categorical__age_cat_1`) was retained for `oblique_random_forest` so random projections do not mix noisy columns into every oblique split.
**Custom class weights (`{0: 1.0, 1: 1.2, 2: 1.1}`):** Initial tests with `class_weight="balanced"` pushed `f1_macro` to `0.54`, but over-predicted `Medium` and `High` so heavily that overall accuracy fell to `0.58–0.59` (below the `0.63` target). Conversely, unweighted models (`None`) reached `0.63` accuracy but stalled at `0.51–0.53` `f1_macro`. Applying a mild class-weight nudge (`{0: 1.0, 1: 1.2, 2: 1.1}`) to the models that support `class_weight` (`logistic_regression`, `hist_gradient_boosting`, `random_forest`, `oblique_random_forest`, `decision_tree`) balanced minority-class recall against overall accuracy.

### Results (ROC-AUC OvR, Accuracy, and Macro F1, development set, 5,771 rows)

**Target thresholds to reach in the same run:** `roc_auc_ovr >= 0.75`, `accuracy >= 0.63`, `f1_macro >= 0.53`.

| Model | Feature selection | Default `roc_auc_ovr` | Tuned `roc_auc_ovr` (mean ± std) | Tuned `accuracy` (CV / OOF) | Tuned `f1_macro` (CV / OOF) | Tuned `balanced_accuracy` | Tuning gain | Optimism | Default gap | Tuned gap | Reaches all 3 targets? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | off | 0.756 | **0.764 ± 0.009** | 0.630 ± 0.004 (0.63) | 0.525 ± 0.009 (0.53) | 0.522 ± 0.007 | +0.008 | ≈0.000 | +0.042 | +0.016 | Borderline (`f1_macro` 0.525 CV / 0.53 OOF) |
| Gradient boosting | off | 0.759 | 0.763 ± 0.009 | 0.628 ± 0.009 (0.63) | 0.530 ± 0.013 (0.53) | 0.524 ± 0.010 | +0.004 | +0.001 | +0.032 | +0.015 | Borderline (`accuracy` 0.628 CV / 0.63 OOF) |
| Hist gradient boosting (`{0:1.0, 1:1.2, 2:1.1}`) | off | 0.737 | 0.762 ± 0.008 | 0.626 ± 0.015 (0.63) | **0.541 ± 0.023 (0.54)** | **0.530 ± 0.019** | +0.025 | +0.001 | +0.085 | +0.016 | Borderline (`accuracy` 0.626 CV / 0.63 OOF) |
| **Logistic regression (`{0:1.0, 1:1.2, 2:1.1}`)** | **off** | **0.760** | **0.761 ± 0.007** | **0.634 ± 0.006 (0.63)** | **0.541 ± 0.013 (0.54)** | 0.528 ± 0.011 | +0.001 | ≈0.000 | +0.003 | +0.003 | **Yes (all 3 in both CV & OOF)** |
| Random forest (`{0:1.0, 1:1.2, 2:1.1}`) | off | 0.729 | 0.760 ± 0.008 | 0.623 ± 0.007 (0.62) | 0.533 ± 0.008 (0.53) | 0.524 ± 0.007 | +0.031 | ≈0.000 | +0.120 | +0.019 | No (`accuracy` 0.623) |
| Logistic regression (`{0:1.0, 1:1.2, 2:1.1}`) | `min_votes 1` | 0.751 | 0.759 ± 0.008 | 0.631 ± 0.006 (0.63) | 0.534 ± 0.015 (0.53) | 0.522 ± 0.013 | +0.008 | -0.001 | ≈0.000 | +0.002 | Yes (`0.759 / 0.631 / 0.534`) |
| Oblique random forest (`{0:1.0, 1:1.2, 2:1.1}`) | `min_votes 1` | 0.709 | 0.758 ± 0.010 | 0.622 ± 0.013 (0.62) | 0.538 ± 0.019 (0.54) | 0.529 ± 0.018 | +0.049 | -0.003 | +0.121 | +0.014 | No (`accuracy` 0.622) |
| AdaBoost | off | 0.748 | 0.758 ± 0.006 | 0.627 ± 0.011 (0.63) | 0.524 ± 0.014 (0.52) | 0.516 ± 0.011 | +0.010 | +0.003 | +0.009 | +0.017 | No (`f1_macro` 0.524) |
| Decision tree (`{0:1.0, 1:1.2, 2:1.1}`) | off | 0.614 | 0.743 ± 0.010 | 0.618 ± 0.009 (0.62) | 0.528 ± 0.017 (0.53) | 0.522 ± 0.013 | +0.129 | +0.003 | +0.111 | +0.019 | No |
| Naive Bayes | off | 0.717 | 0.738 ± 0.011 | 0.596 ± 0.007 (0.60) | 0.431 ± 0.011 (0.43) | 0.442 ± 0.009 | +0.021 | -0.001 | +0.002 | ≈0.000 | No |
| Dummy (floor) | `min_votes 2` | 0.500 | – | 0.540 ± 0.001 (0.54) | 0.234 ± 0.000 (0.23) | 0.333 ± 0.000 | – | – | +0.000 | – | No |

Chosen hyperparameters (tuning on the whole development set):

| Model | Hyperparameters |
|---|---|
| XGBoost | n_estimators=152, learning_rate=0.211, max_depth=4, min_child_weight=5.47, subsample=0.53, colsample_bytree=0.60, reg_lambda=0.015, reg_alpha=0.20, gamma=2.85 |
| Gradient boosting | n_estimators=34, learning_rate=0.052, max_depth=4, min_samples_leaf=76, subsample=0.53 |
| Hist gradient boosting | max_iter=120, learning_rate=0.047, max_leaf_nodes=6, min_samples_leaf=123, l2=0.551, fixed `class_weight={0: 1.0, 1: 1.2, 2: 1.1}` |
| **Logistic regression (`off`)** | **C=0.079, fixed `class_weight={0: 1.0, 1: 1.2, 2: 1.1}`, median imputation, no scaling, ordinal encoding** |
| Random forest | max_depth=16, min_samples_leaf=31, max_features=0.5, fixed `class_weight={0: 1.0, 1: 1.2, 2: 1.1}` |
| Logistic regression (`min_votes 1`) | C=3.97, fixed `class_weight={0: 1.0, 1: 1.2, 2: 1.1}`, mean imputation, standard scaling, ordinal encoding, min_votes=1 |
| Oblique random forest | max_depth=12, min_samples_leaf=130, n_components=10, fixed `class_weight={0: 1.0, 1: 1.2, 2: 1.1}`, min_votes=1 |
| AdaBoost | base tree max_depth=4, n_estimators=72, learning_rate=0.139 |
| Decision tree | max_depth=18, min_samples_leaf=149, criterion=entropy, fixed `class_weight={0: 1.0, 1: 1.2, 2: 1.1}` |
| Naive Bayes | var_smoothing=0.091 |

**Interpretation.** The dummy baseline predicts `"Low"` for every person and scores 0.500 `roc_auc_ovr`, 0.540 accuracy (matching the 54.0% share of `"Low"` in the development set, 3,118 of 5,771), and 0.234 `f1_macro` (since F1 is 0.70 for `"Low"` and 0.00 for `"Medium"` and `"High"`). Every model learns substantial signal above this floor (+0.24 to +0.26 on `roc_auc_ovr`), and ranks COMPAS's own risk categories better (~0.764 `roc_auc_ovr`) than it ranked actual two-year recidivism in Weeks 7 and 8 (~0.727 `roc_auc_ovr`). That makes sense: `score_text` is generated by a deterministic scoring formula over age and criminal history, whereas real-world recidivism has far more irreducible noise.

Just as in Weeks 7 and 8, the top eight model configurations (`xgboost` 0.764, `gradient_boosting` 0.763, `hist_gradient_boosting` 0.762, `logistic_regression` 0.761, `random_forest` 0.760, `oblique_random_forest` 0.758, and `adaboost` 0.758) span only 0.006 `roc_auc_ovr`—smaller than a single fold-level standard deviation (0.006 to 0.010)—so they are statistically tied on ranking ability. Tuning again helps the models that overfit by default the most: the decision tree (+0.129), oblique random forest (+0.049), random forest (+0.031), and hist gradient boosting (+0.025) all see large gains because tuning shrinks their train-validation gaps from +0.085–0.121 down to +0.014–0.019 (via large leaf sizes of 31 to 149 rows, or 6-leaf trees in hist gradient boosting). By contrast, `logistic_regression` is already well regularized by default (`0.760` untuned vs. `0.761` tuned, with a `+0.003` train-validation gap). The optimism check is near zero (-0.003 to +0.003) across every model, confirming nested CV is not inflating the results. Fold 4 remains the hardest fold across all models (`roc_auc_ovr` 0.733 to 0.757) and Fold 5 the easiest (`0.753` to `0.778`).

### Out-of-fold per-class performance and fairness (most likely class, `argmax`)

| Model | Class weighting | Low F1 (Prec / Rec) | Medium F1 (Prec / Rec) | High F1 (Prec / Rec) | Macro F1 (OOF / CV) | Accuracy (OOF / CV) | False High: African-American | False High: Caucasian | Ratio |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic regression (`off`)** | **`{0:1.0, 1:1.2, 2:1.1}`** | **0.78 (0.70 / 0.87)** | **0.38 (0.42 / 0.35)** | **0.47 (0.64 / 0.37)** | **0.54 (0.541)** | **0.63 (0.634)** | **0.07** | **0.03** | **2.33x** |
| Hist gradient boosting | `{0:1.0, 1:1.2, 2:1.1}` | 0.77 (0.70 / 0.85) | 0.37 (0.40 / 0.33) | 0.49 (0.60 / 0.41) | **0.54 (0.541)** | 0.63 (0.626) | 0.10 | 0.04 | 2.50x |
| Oblique random forest (`min_votes 1`) | `{0:1.0, 1:1.2, 2:1.1}` | 0.77 (0.71 / 0.84) | 0.36 (0.39 / 0.33) | 0.49 (0.59 / 0.42) | **0.54 (0.538)** | 0.62 (0.622) | 0.10 | 0.05 | 2.00x |
| Logistic regression (`min_votes 1`) | `{0:1.0, 1:1.2, 2:1.1}` | 0.78 (0.70 / 0.88) | 0.37 (0.42 / 0.33) | 0.46 (0.65 / 0.36) | 0.53 (0.534) | 0.63 (0.631) | 0.07 | 0.03 | 2.33x |
| Random forest | `{0:1.0, 1:1.2, 2:1.1}` | 0.77 (0.70 / 0.85) | 0.35 (0.40 / 0.31) | 0.48 (0.59 / 0.40) | 0.53 (0.533) | 0.62 (0.623) | 0.09 | 0.05 | 1.80x |
| Decision tree | `{0:1.0, 1:1.2, 2:1.1}` | 0.77 (0.70 / 0.85) | 0.34 (0.40 / 0.30) | 0.48 (0.55 / 0.42) | 0.53 (0.528) | 0.62 (0.618) | 0.11 | 0.06 | 1.83x |
| Gradient boosting | None | 0.77 (0.69 / 0.88) | 0.31 (0.40 / 0.26) | 0.50 (0.59 / 0.43) | 0.53 (0.530) | 0.63 (0.628) | 0.10 | 0.05 | 2.00x |
| XGBoost | None | 0.77 (0.68 / 0.89) | 0.29 (0.41 / 0.23) | **0.51** (0.59 / **0.44**) | 0.53 (0.525) | 0.63 (0.630) | 0.10 | 0.05 | 2.00x |
| AdaBoost | None | 0.77 (0.68 / 0.89) | 0.32 (0.40 / 0.27) | 0.48 (0.61 / 0.39) | 0.52 (0.524) | 0.63 (0.627) | 0.08 | 0.04 | 2.00x |
| Naive Bayes | None | 0.75 (0.62 / 0.95) | 0.19 (0.35 / 0.13) | 0.35 (0.63 / 0.25) | 0.43 (0.431) | 0.60 (0.596) | 0.05 | 0.03 | 1.67x |
| Dummy (floor) | None | 0.70 (0.54 / 1.00) | 0.00 (0.00 / 0.00) | 0.00 (0.00 / 0.00) | 0.23 (0.234) | 0.54 (0.540) | 0.00 | 0.00 | – |

**Interpretation: Operating points, class weights, and why Logistic Regression wins overall.** While `xgboost` (`0.764`) and `gradient_boosting` (`0.763`) sit fractionally ahead on `roc_auc_ovr`, neither supports `class_weight` directly in `params`. Under unweighted `argmax` classification, both models achieve `0.630 / 0.628` accuracy by predicting `"Low"` heavily (`recall = 0.89 / 0.88`), which leaves `"Medium"` recall at `0.23–0.26` (`F1 = 0.29–0.31`) and holds `f1_macro` at `0.525–0.530` (right on the boundary of the `0.53` target).

By contrast, adding the mild custom weights `{0: 1.0, 1: 1.2, 2: 1.1}` to `logistic_regression` (with `feature_selection: off`, run `20261113_095946`) lifts `"Medium"` recall to `0.35` (`F1 = 0.38`) while keeping `"Low"` recall at `0.87`. As a result, **`logistic_regression` is the only model that cleanly clears all three targets in both nested CV (`roc_auc_ovr = 0.761`, `accuracy = 0.634`, `f1_macro = 0.541`) and out-of-fold evaluation (`0.63` accuracy, `0.54` Macro F1)**, while also having the smallest train-validation gap (`+0.003`) and simplest model structure. `hist_gradient_boosting` with the same weights (`20261113_113246`) is a close runner-up (`roc_auc_ovr = 0.762`, `f1_macro = 0.541`, `accuracy = 0.626` CV / `0.63` OOF).

**Fairness (False High rate by race).** In the three-class task, fairness is measured as the share of defendants who were **not** truly `"High"` (true `"Low"` or `"Medium"`, `n=2,098` African-American and `n=1,698` Caucasian in the development set) but were falsely predicted as `"High"` by our model. Across every model, African-American defendants are falsely placed into the `"High"` category at roughly **1.8x to 2.5x** the rate of Caucasian defendants (`0.07 vs. 0.03` for the winning `logistic_regression` model; `0.08–0.11 vs. 0.04–0.06` across the boosting and forest models). Even though `race` is excluded from the input features, the models learn the demographic proxy correlations embedded in `age` and `priors_count` that drive COMPAS's own High-risk assignments. Notably, `logistic_regression` achieves both the best target metric balance and the lowest absolute False High rate (`0.07` for African-American, `0.03` for Caucasian) among all competitive models.

### Discussion & Reflections

The hardest hidden binary assumption to find was inside `VoteSelector.fit()`, where two separate assumptions were stacked together: `solver="liblinear"` crashed on three classes in modern scikit-learn (fixed by switching to `solver="saga"`), while `np.abs(lasso.coef_[0]) > 1e-8` would have silently checked only the L1 coefficients for Class 0 (`Low`) and ignored features used to separate `Medium` and `High` (fixed with `.any(axis=0)`). Reaching all three target scores simultaneously (`roc_auc_ovr >= 0.75`, `accuracy >= 0.63`, and `f1_macro >= 0.53`) came down to managing the operating-point trade-off between overall accuracy and minority-class F1: unweighted tree boosters (`xgboost` 0.764, `gradient_boosting` 0.763) excelled at ranking and accuracy (`0.63`) but under-predicted `Medium` (`f1_macro = 0.525–0.530`), whereas `class_weight="balanced"` lifted `f1_macro` (`0.54`) at the cost of dropping accuracy to `0.59`. Applying mild custom weights (`{0: 1.0, 1: 1.2, 2: 1.1}`) to `logistic_regression` (with `feature_selection: off`) solved both at once (`roc_auc_ovr = 0.761`, `accuracy = 0.634`, `f1_macro = 0.541`). Across every single model, **Medium** (`decile_score` 5–7) remains by far the hardest class to predict, topping out at `0.38` F1 (compared to `0.77–0.78` for `Low` and `0.46–0.51` for `High`). Because `Medium` is an intermediate bucket squeezed between the majority `Low` class (54%) and the extreme `High` class (20%) along only a few coarse features (`age` and `priors_count`), its feature distribution heavily overlaps with both neighbors, causing more than half of true `Medium` defendants (`810` of `1,531` in the winning model) to be predicted as `Low` and `188` as `High`.

# Week 10: missing values and predicting new data

## Changes

### Missing values

Gaps in categorical columns are filled with the most frequent value. For numeric columns, `preprocessing.imputation.numeric_strategy` chooses how:

| `numeric_strategy` | Fills a gap with | Notes |
|---|---|---|
| `median` / `mean` | the column's median / mean on the training rows | fast baselines |
| `knn` | the mean of the `n_neighbors` most similar rows | needs scaled columns (the pipeline scales first) |
| `iterative` | a prediction from the other numeric columns, refined over `max_iter` rounds (MICE, one completed dataset) | |
| `random_forest` | the same, with a random forest per column (missForest) | the slowest |

All of them are fitted on the training rows of each fold, inside the Pipeline, and none uses the target, so the same imputer runs on new data in `predict.py`. They only see the numeric columns. The `<col>_was_missing` flags (`mnar_indicator_sources`) keep the fact that a value was missing, for the columns where that is informative.

### Predicting new data

At the end of every run, `main.py` saves the final model (the whole fitted Pipeline: imputers, encoders, selected features and model, refit on all development rows) to `models/model.joblib` (`output.model_path` in `config.yaml`; each run overwrites it). To predict a CSV with the same columns as the training data (the target isn't needed):

```bash
python main.py                                                   # train, evaluate, save models/model.joblib
python predict.py data/compas_new_people.csv predictions.csv     # predict
```

The output has one row per input row: the id, the probability of reoffending within two years, and the prediction (1 if the probability is at least 0.5). The new file is cleaned with the rules in `config.yaml` and never de-duplicated: every row gets a prediction, including rows with gaps, placeholder values, categories not seen in training, or a person repeated twice. For the three-class model, add `--config config_multiclass.yaml`: the output then has one probability per class and the most likely class.

## Results

The five imputers were compared with the same model and settings: `hist_gradient_boosting` with its default hyperparameters (`random_state: 42`, `class_weight: balanced`), tuning off, and the same 5 CV folds for every run. The p-values come from `corrected_paired_test()` on the per-fold ROC-AUC, against `median`.

| Imputer | roc_auc: mean ± std | vs median: difference | folds won | p-value |
|---|---|---|---|---|
| median | 0.717 ± 0.020 | n/a | n/a | n/a |
| knn | 0.716 ± 0.018 | -0.0008 | 2/5 | 0.55 |
| mean | 0.716 ± 0.019 | -0.0009 | 3/5 | 0.72 |
| iterative | 0.716 ± 0.016 | -0.0013 | 3/5 | 0.75 |
| random_forest | 0.716 ± 0.019 | -0.0004 | 3/5 | 0.91 |

All five imputers score within 0.0013 ROC-AUC of `median` (0.716-0.717), the corrected paired test gives p between 0.55 and 0.91, and every difference is much smaller than the fold-to-fold std (0.016-0.020), so I found no evidence that a more elaborate imputer helps. I keep `median`: it is the simplest and cheapest, and `knn`, `iterative` and `random_forest` fit more and give nothing back. This is plausible because few values are missing: on the development rows, about 7% of `priors_count`, 3% of `juv_fel_count` and 2% of `age`.

# Week 11: outliers

## Invalid, extreme, anomalous

Invalid values (age 150, negative priors) were already turned into gaps by `validity_rules`. Extreme values are kept: the 6 people with 30+ priors are adults aged 39–61 with long records, not errors, and a deployed model can't delete a person. So the question is not how to remove outliers but how unusual each person is, and whether that helps the model.

## The `AnomalyScore` step

A learned step of the Pipeline, right after `"prep"` (so its input is already imputed, scaled and encoded) and before `"select"` (so feature selection can keep or drop it). It is fitted on the training rows of each fold and scores any rows, so it never sees the validation fold and runs unchanged in `predict.py`.

```yaml
anomaly:
  method: "none"          # none | isolation_forest | lof
  n_neighbors: 20         # lof: larger than the largest group of identical rows (14 here)
  random_state: 42        # isolation_forest
```

- `isolation_forest`: the score is `-score_samples(X)`, about 0.5 for ordinary rows and close to 1 for anomalies. No scaling needed.
- `lof`: `novelty=True`, so new people can be scored; on the training rows the step uses `negative_outlier_factor_`, so a row is not counted as its own neighbour. `n_neighbors` must be larger than the largest group of identical rows (14 on COMPAS): with k = 5 the scores explode to hundreds of millions.


## Results

**Are the unusual people the ones who reoffend?** `python check_anomalies.py` (development rows, top 5% by score):

| Method | People in top 5% | Reoffend rate, top 5% | Reoffend rate, rest | Rank correlation score vs reoffended |
|---|---|---|---|---|
| isolation_forest | 289 | 0.561 | 0.445 | 0.048 |
| lof | 294 | 0.548 | 0.445 | 0.069 |

The most unusual people reoffend somewhat more often (56% against 45%), but over all the rows the score barely ranks who reoffends (Spearman 0.05–0.07).

**Does it help the model?** `hist_gradient_boosting` with its default hyperparameters (`random_state: 42`, `class_weight: balanced`), KNN imputer, feature selection off, tuning off, the same 5 CV folds for each run. p-values from `python compare_imputers.py --by anomaly --baseline none` (corrected paired t-test on the per-fold ROC-AUC):

| Anomaly | roc_auc: mean ± std | vs none: difference | folds won | p-value |
|---|---|---|---|---|
| none | 0.716 ± 0.018 | n/a | n/a | n/a |
| isolation_forest | 0.708 ± 0.022 | -0.0077 | 1/5 | 0.30 |
| lof | 0.711 ± 0.019 | -0.0050 | 1/5 | 0.24 |

**With tuning on** (the best configuration: the same model, KNN imputer, feature selection off, Optuna with 30 trials, ROC-AUC from nested CV on the same 5 outer folds). `python compare_imputers.py --by anomaly --baseline none --tuned true`:

| Anomaly | roc_auc: mean ± std (nested CV) | vs none: difference | folds won | p-value |
|---|---|---|---|---|
| none | 0.728 ± 0.015 | n/a | n/a | n/a |
| isolation_forest | 0.725 ± 0.013 | -0.0035 | 1/5 | 0.25 |
| lof | 0.725 ± 0.017 | -0.0031 | 1/5 | 0.25 |

Tuning closes most of the gap (from about -0.008 / -0.005 untuned to about -0.003), but the score still doesn't help: no score wins on 4 of 5 folds for both methods. With feature selection on, the selector drops the Isolation Forest score and keeps the LOF one.

## Conclusions

The anomaly score does not help on COMPAS, with or without tuning: both methods are slightly below no score and win only 1 of 5 folds, and neither difference is bigger than fold-to-fold noise. The best model is still tuned `hist_gradient_boosting` without the score (0.728). A likely reason: gradient boosting already finds the rare combinations (many juvenile records with many priors) by splitting on the columns themselves, so a score built from the same columns adds little information and some noise. `method` stays `"none"` in `config.yaml`.
