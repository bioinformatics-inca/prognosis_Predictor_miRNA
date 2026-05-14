"""
================================================================================
Author      : Cristiane Esteves teixeira, M.Sc
Affiliation : Instituto Nacional de Câncer (INCA)
E-mail      : cristiane.esteves@ensino.inca.gov.br
================================================================================

================================================================================
ml_pipeline_miRNA_clinical.py
================================================================================
Machine Learning pipeline for ovarian cancer prognosis classification,
combining miRNA expression biomarkers with clinical variables.

Overall workflow:
  1. Load and preprocess datasets (training cohort + external validation)
  2. Recursive Feature Elimination (RFE) over a range of 3–13 features
  3. Hyperparameter tuning (GridSearchCV) across 8 distinct classifiers
  4. Probability calibration via Platt scaling (CalibratedClassifierCV)
  5. Threshold tuning to guarantee a minimum recall of 83 %
  6. Evaluation on internal hold-out and two external validation sets
     (full cohort and FIGO stages II–IV), with 95 % CI via bootstrap
     and Clopper-Pearson
  7. Export results to CSV and serialize the best model to .pkl

Main dependencies:
  scikit-learn, imbalanced-learn, lightgbm, xgboost, tabpfn, scipy, joblib
================================================================================
"""

import os
import pandas as pd
import numpy as np
import warnings
import scipy.stats as stats
from sklearn.utils import resample

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score, RepeatedStratifiedKFold,
               recall_score, confusion_matrix, f1_score, brier_score_loss)

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from tabpfn import TabPFNClassifier

import joblib

warnings.filterwarnings("ignore")

# ── Global constants ──────────────────────────────────────────────────────────
SEED = 42             # Random seed for full reproducibility
TARGET_RECALL = 0.83  # Minimum sensitivity (recall) required at threshold tuning
N_BOOTSTRAP = 5000    # Number of bootstrap iterations (used in some helper calls)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def bootstrap_metric(y_true, y_pred=None, y_prob=None, metric="auc", n_iter=2000, seed=42):
    """
    Estimate a classification metric with 95 % confidence interval via
    non-parametric bootstrap resampling.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like, optional
        Hard class predictions (required for recall, balacc, f1, vpn).
    y_prob : array-like, optional
        Predicted probabilities for the positive class (required for auc, brier).
    metric : str
        One of {"auc", "recall", "balacc", "f1", "vpn", "brier"}.
    n_iter : int
        Number of bootstrap resamples.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    mean, sd, ci_low, ci_high : float
        Bootstrap mean, standard deviation, 2.5th and 97.5th percentiles.
    """

    rng = np.random.RandomState(seed)
    scores = []

    y_true = np.array(y_true)

    if y_pred is not None:
        y_pred = np.array(y_pred)

    if y_prob is not None:
        y_prob = np.array(y_prob)

    for _ in range(n_iter):

        # Draw a bootstrap sample with replacement
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)

        yt = y_true[idx]

        # Skip samples that contain only one class (metric undefined)
        if len(np.unique(yt)) < 2:
            continue

        if metric == "auc":
            yp = y_prob[idx]
            score = roc_auc_score(yt, yp)

        elif metric == "recall":
            yp = y_pred[idx]
            score = recall_score(yt, yp)

        elif metric == "balacc":
            yp = y_pred[idx]
            score = balanced_accuracy_score(yt, yp)

        elif metric == "f1":
            yp = y_pred[idx]
            score = f1_score(yt, yp)

        elif metric == "vpn":
            yp = y_pred[idx]
            score = vpn_score(yt, yp)

        elif metric == "brier":
            yp = y_prob[idx]
            score = brier_score_loss(yt, yp)

        scores.append(score)

    scores = np.array(scores)

    mean = np.mean(scores)
    sd = np.std(scores, ddof=1)

    ci_low = np.percentile(scores, 2.5)
    ci_high = np.percentile(scores, 97.5)

    return mean, sd, ci_low, ci_high


def clopper_pearson(x, n, conf_level=0.95):
    """
    Compute an exact binomial (Clopper-Pearson) confidence interval.

    Preferred over normal approximation for small samples or extreme
    proportions (e.g., sensitivity in a small positive class).

    Parameters
    ----------
    x : int
        Number of successes (e.g., true positives).
    n : int
        Total number of trials (e.g., all positives).
    conf_level : float
        Desired coverage (default 0.95 -> 95 % CI).

    Returns
    -------
    lower, upper : float
        Lower and upper bounds of the confidence interval.
    """

    if n == 0:
        return 0.0, 0.0

    alpha = 1 - conf_level
    lower = stats.beta.ppf(alpha / 2, x, n - x + 1) if x > 0 else 0.0
    upper = stats.beta.ppf(1 - alpha / 2, x + 1, n - x) if x < n else 1.0

    return lower, upper


def get_auc_bootstrap(y_true, y_probs, n_iterations=2000, seed=42):
    """
    Compute AUC-ROC with a 95 % bootstrap confidence interval.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_probs : array-like
        Predicted probabilities for the positive class.
    n_iterations : int
        Number of bootstrap resamples.
    seed : int
        Random state for reproducibility.

    Returns
    -------
    auc_mean : float
        AUC computed on the full (non-resampled) dataset.
    ci_low, ci_high : float
        2.5th and 97.5th bootstrap percentiles.
    """

    rng = np.random.RandomState(seed)
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    scores = []

    for i in range(n_iterations):

        idx = rng.randint(0, len(y_true), len(y_true))

        if len(np.unique(y_true[idx])) < 2:
            continue

        scores.append(roc_auc_score(y_true[idx], y_probs[idx]))

    scores = np.array(scores)

    auc_mean = roc_auc_score(y_true, y_probs)
    ci_low = np.percentile(scores, 2.5)
    ci_high = np.percentile(scores, 97.5)

    return auc_mean, ci_low, ci_high


def vpn_score(y_true, y_pred):
    """
    Compute the Negative Predictive Value (NPV).

    NPV = TN / (TN + FN)
    Measures the probability that a negative prediction is truly negative.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Hard class predictions.

    Returns
    -------
    float
        NPV score, or 0 if the denominator is zero.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    return tn / (tn + fn) if (tn + fn) > 0 else 0


def get_tuned_threshold(y_true, y_probs, min_recall):
    """
    Find the decision threshold that maximises balanced accuracy while
    keeping recall >= min_recall.

    Iterates over 1 000 evenly-spaced thresholds in [0, 1] and selects
    the one with the highest balanced accuracy among all thresholds that
    satisfy the minimum recall constraint.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_probs : array-like
        Predicted probabilities for the positive class.
    min_recall : float
        Minimum acceptable recall (sensitivity).

    Returns
    -------
    float
        Optimal threshold value.
    """

    best_t = 0.5
    best_bal = 0

    for t in np.linspace(0, 1, 1000):

        pred = (y_probs >= t).astype(int)

        rec = recall_score(y_true, pred)

        if rec >= min_recall:

            bal = balanced_accuracy_score(y_true, pred)

            if bal > best_bal:
                best_bal = bal
                best_t = t

    return best_t


def calibration_metrics_advanced(y_true, probs):
    """
    Assess probability calibration via logistic calibration regression
    (Cox's calibration framework).

    A perfectly calibrated model yields intercept ~ 0 and slope ~ 1.
    The logit of the predicted probabilities is used as the single
    predictor; the fitted intercept and slope are the calibration metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    probs : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    intercept : float
        Calibration-in-the-large (overall bias).
    slope : float
        Calibration slope (refinement / sharpness).
    se_slope : float
        Standard error of the slope estimate.
    ci_slope : str
        Formatted 95 % Wald CI for the slope, e.g. "[0.812-1.134]".
    """

    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)  # Avoid log(0)

    # Convert probabilities to log-odds (logit scale)
    logit_p = np.log(probs / (1 - probs))

    # Fit logistic regression of observed outcomes on predicted logit
    model = LogisticRegression(solver="lbfgs")
    model.fit(logit_p.reshape(-1, 1), y_true)

    slope = model.coef_[0][0]
    intercept = model.intercept_[0]

    # Re-predict calibrated probabilities for SE calculation
    pred_p = model.predict_proba(logit_p.reshape(-1, 1))[:, 1]

    # Variance weights for the Fisher information matrix
    v = pred_p * (1 - pred_p)

    X = np.column_stack([np.ones(len(logit_p)), logit_p])

    try:
        # SE of slope via inverse Fisher information
        se_slope = np.sqrt(np.linalg.inv(np.dot(X.T * v, X))[1, 1])
        ci_slope = f"[{slope - 1.96 * se_slope:.3f}-{slope + 1.96 * se_slope:.3f}]"
    except:
        se_slope = np.nan
        ci_slope = "NaN"

    return intercept, slope, se_slope, ci_slope


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

# Load training cohort (internal dataset with miRNA + clinical features)
df_tr = pd.read_csv('/data/top10_tcga.csv', index_col=0)

# Load external validation cohort
df_ex = pd.read_csv('/data/PCR_inca.csv', index_col=0)

# Log2-transform MUC16 (CA-125) in the external cohort if present,
# to match the scale used during training
if 'MUC16' in df_ex.columns:
    df_ex['MUC16'] = np.log2(df_ex['MUC16'] + 1)

# Encode target variable as binary: Poor -> 1 (event), Good -> 0
df_tr['progn'] = df_tr['progn'].map({'Poor': 1, 'Good': 0})
df_ex['ClassProg'] = df_ex['ClassProg'].map({'Poor': 1, 'Good': 0})

# Subset of external cohort restricted to FIGO stages III-IV
df_ex_filt = df_ex[df_ex['figo_stage'].isin([3, 4])]


# =============================================================================
# FEATURE DEFINITION
# =============================================================================

# Identify miRNA columns by searching for known miRNA ID substrings
mirna_patterns = ['187', '149', '205', '150', '143', '485', '200a', '23a', '106b', '151a']
mirna_cols = [c for c in df_tr.columns if any(p in c for p in mirna_patterns)]

# Clinical covariates included alongside miRNA expression
clinical_cols = ['age_at_index', 'figo_stage', 'MUC16']

features = mirna_cols + clinical_cols

X = df_tr[features]
y = df_tr['progn']


# =============================================================================
# TRAIN / INTERNAL HOLD-OUT SPLIT
# =============================================================================

# 90 % training, 10 % hold-out; stratified to preserve class balance
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    stratify=y,
    random_state=SEED
)


# =============================================================================
# EXTERNAL VALIDATION SETS — SCALE CORRECTION
# =============================================================================

# Full external cohort and FIGO-filtered subset
X_ex = df_ex[features]
X_ex_filt = df_ex_filt[features]

# Columns that correspond to miRNA features
cols_mir = [c for c in features if 'mir' in c.lower()]

# Back-transform miRNA values from -delta-Ct to relative expression (2^(-delta-Ct))
# so the external cohort is on the same numeric scale as the training data
X_rel_full = X_ex.copy()
X_rel_full[cols_mir] = 2**(-X_ex[cols_mir])

X_rel_filt = X_ex_filt.copy()
X_rel_filt[cols_mir] = 2**(-X_ex_filt[cols_mir])


# =============================================================================
# MODEL DEFINITIONS & HYPERPARAMETER GRIDS
# =============================================================================

# Each entry defines a classifier and its candidate hyperparameter space.
# All models also receive the shared base_param_grid (scaler + sampler) below.
models_config = {

    'LogReg': {
        'model': LogisticRegression(solver='liblinear', random_state=SEED),
        'params': {
            'classifier__C': [0.01, 0.1, 1, 10, 0.001],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__class_weight': [None, 'balanced']
        }
    },

    'SVC': {
        'model': SVC(probability=True, random_state=SEED),
        'params': {
            'classifier__C': [0.1, 1, 10, 0.001],
            'classifier__kernel': ['linear', 'rbf'],
            'classifier__class_weight': [None, 'balanced']
        }
    },

    'ExtraTrees': {
        'model': ExtraTreesClassifier(random_state=SEED),
        'params': {
            'classifier__n_estimators': [300, 200],
            'classifier__max_depth': [None, 5, 3, 4],
            'classifier__class_weight': [None, 'balanced']
        }
    },

    'RandomForest': {
        'model': RandomForestClassifier(random_state=SEED),
        'params': {
            'classifier__n_estimators': [200, 300],
            'classifier__max_depth': [None, 5],
            'classifier__class_weight': [None, 'balanced']
        }
    },

    'LightGBM': {
        'model': LGBMClassifier(random_state=SEED, verbosity=-1),
        'params': {
            'classifier__n_estimators': [200, 300],
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__num_leaves': [7, 15],
            'classifier__max_depth': [3, 5]
        }
    },

    'XGBoost': {
        'model': XGBClassifier(
            random_state=SEED,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        ),
        'params': {
            'classifier__n_estimators': [200, 300],
            'classifier__learning_rate': [0.01, 0.05],
            'classifier__max_depth': [3, 4]
        }
    },

    'MLP': {
        'model': MLPClassifier(max_iter=2000, early_stopping=True, random_state=SEED),
        'params': {
            'classifier__hidden_layer_sizes': [(10,), (20,), (10, 5)],
            'classifier__activation': ['relu', 'tanh']
        }
    },

    # TabPFN: prior-fitted network trained on meta-learning — no scaling or resampling needed
    'TabPFN': {
        'model': TabPFNClassifier(device='cpu', seed=SEED),
        'params': {}
    }
}

# Shared grid applied to every model: preprocessing scaler and class-imbalance sampler
base_param_grid = {
    'scaler': [RobustScaler(), StandardScaler(), 'passthrough'],
    'sampler': [
        'passthrough',
        SMOTE(random_state=SEED),
        RandomUnderSampler(random_state=SEED),
        RandomOverSampler(random_state=SEED)
    ]
}


# =============================================================================
# MAIN TRAINING LOOP — RFE x MODEL x HYPERPARAMETER SEARCH
# =============================================================================

results = []  # Accumulates one row per (N_features, model) combination

# Track the globally best model across all iterations (ranked by internal AUC)
best_global_auc = -1
best_global_model = None
best_global_info = None

for N_FEATURES in range(3, 14):  # Sweep number of selected features from 3 to 13

    print("\n============================")
    print("RFE FEATURES:", N_FEATURES)
    print("============================")

    # RFE uses a sparse logistic regression as the feature ranking estimator
    rfe_selector = RFE(
        estimator=LogisticRegression(
            solver='liblinear',
            random_state=SEED,
            C=0.01),
        n_features_to_select=N_FEATURES,
        step=1
    )

    for name, config in models_config.items():

        print("---", name)

        # Build the full pipeline: scaler -> sampler -> RFE -> classifier
        pipe = Pipeline([
            ('scaler', 'passthrough'),
            ('sampler', 'passthrough'),
            ('rfe', rfe_selector),
            ('classifier', config['model'])
        ])

        # Merge shared and model-specific hyperparameter grids
        current_grid = {**base_param_grid, **config['params']}

        # TabPFN is sensitive to preprocessing; skip scaling and resampling
        if name == 'TabPFN':
            current_grid['scaler'] = ['passthrough']
            current_grid['sampler'] = ['passthrough']

        # Grid search with 5-fold CV; primary scoring = AUC (for selection),
        # refit on balanced accuracy (final estimator chosen)
        grid = GridSearchCV(
            pipe,
            current_grid,
            cv=5,
            scoring='roc_auc',
            refit='balanced_accuracy',
            n_jobs=-1
        )

        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_

        print("Best Params:", grid.best_params_)

        # Retrieve the RFE boolean mask to identify which features were retained
        selector = best_pipe.named_steps['rfe']
        selected_features = X_train.columns[selector.support_]

        print("Selected:", list(selected_features))

        # ── Probability calibration ──────────────────────────────────────────
        # Platt scaling (sigmoid) with 3-fold internal CV to correct
        # overconfident or underconfident raw probabilities
        model_calib = CalibratedClassifierCV(best_pipe, method='sigmoid', cv=3)

        # Out-of-fold predicted probabilities for unbiased threshold tuning
        # (avoids threshold being optimised on the same fold used for fitting)
        cv_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

        oof_probs = cross_val_predict(
            model_calib,
            X_train,
            y_train,
            cv=cv_skf,
            method="predict_proba",
            n_jobs=-1
        )[:, 1]

        # Find the threshold that maximises balanced accuracy at recall >= 83 %
        t_oof = get_tuned_threshold(y_train, oof_probs, TARGET_RECALL)
        print("Threshold otimizado:", t_oof)

        # Refit the calibrated model on the full training set before evaluation
        model_calib.fit(X_train, y_train)

        # ── Internal hold-out evaluation ─────────────────────────────────────
        probs_h = model_calib.predict_proba(X_test)[:, 1]
        preds_h = (probs_h >= t_oof).astype(int)

        # Bootstrap 95 % CIs for all performance metrics
        auc_m,   auc_sd,   auc_low,   auc_high   = bootstrap_metric(y_test, y_prob=probs_h, metric="auc")
        rec_m,   rec_sd,   rec_low,   rec_high   = bootstrap_metric(y_test, y_pred=preds_h, metric="recall")
        bal_m,   bal_sd,   bal_low,   bal_high   = bootstrap_metric(y_test, y_pred=preds_h, metric="balacc")
        f1_m,    f1_sd,    f1_low,    f1_high    = bootstrap_metric(y_test, y_pred=preds_h, metric="f1")
        vpn_m,   vpn_sd,   vpn_low,   vpn_high   = bootstrap_metric(y_test, y_pred=preds_h, metric="vpn")
        brier_m, brier_sd, brier_low, brier_high = bootstrap_metric(y_test, y_prob=probs_h, metric="brier")

        # AUC CI via dedicated function (point estimate on the full hold-out set)
        auc_sd_h, auc_low_h, auc_high_h = get_auc_bootstrap(y_test, probs_h)

        # Exact binomial (Clopper-Pearson) CI for recall on the hold-out
        n1_h = int(sum(y_test))
        tp_h = int(sum((y_test == 1) & (preds_h == 1)))
        rec_low_h, rec_high_h = clopper_pearson(tp_h, n1_h)

        # Point estimates on the hold-out set
        auc_int = roc_auc_score(y_test, probs_h)

        # Update the global best model tracker
        if auc_int > best_global_auc:
            best_global_auc = auc_int
            best_global_model = model_calib
            best_global_info = {
                "Model": name,
                "N_features": N_FEATURES,
                "Selected_features": list(selected_features),
                "Threshold": t_oof
            }

        rec_int   = recall_score(y_test, preds_h)
        bal_int   = balanced_accuracy_score(y_test, preds_h)
        f1_int    = f1_score(y_test, preds_h)
        vpn_int   = vpn_score(y_test, preds_h)
        brier_int = brier_score_loss(y_test, probs_h)

        # Brier Skill Score: improvement over a no-information (prevalence) baseline
        prevalence_poor = y_train.mean()
        brier_baseline  = prevalence_poor * (1 - prevalence_poor)
        bss_int = 1 - (brier_int / brier_baseline)

        # Calibration metrics for the internal hold-out
        cal_intercept_int, cal_slope_int, slope_sd_int, slope_ci_int = calibration_metrics_advanced(y_test, probs_h)

        print("\n--- INTERNAL HOLD-OUT")
        print("AUC:", auc_int, f"IC95% [{auc_low_h:.3f}-{auc_high_h:.3f}]")
        print("Recall:", rec_int, f"IC95% [{rec_low_h:.3f}-{rec_high_h:.3f}]")
        print("BalAcc:", bal_int)
        print("F1:", f1_int)
        print("VPN:", vpn_int)
        print("Brier:", brier_int)
        print("Brier Skill Score:", bss_int)
        print("Calibration Intercept:", cal_intercept_int)
        print("Calibration Slope:", cal_slope_int)
        print("CM\n", confusion_matrix(y_test, preds_h))

        # Build the result row for this (N_features, model) combination
        row = {
            "Model": name,
            "N_features": N_FEATURES,
            "Features": ", ".join(selected_features),
            "Best_Params": str(grid.best_params_),
            "Threshold": t_oof,
            "AUC_int":    f"{auc_m:.3f} ± {auc_sd:.3f} [{auc_low:.3f}-{auc_high:.3f}]",
            "Recall_int": f"{rec_m:.3f} ± {rec_sd:.3f} [{rec_low:.3f}-{rec_high:.3f}]",
            "BalAcc_int": f"{bal_m:.3f} ± {bal_sd:.3f} [{bal_low:.3f}-{bal_high:.3f}]",
            "F1_int":     f"{f1_m:.3f} ± {f1_sd:.3f} [{f1_low:.3f}-{f1_high:.3f}]",
            "VPN_int":    f"{vpn_m:.3f} ± {vpn_sd:.3f} [{vpn_low:.3f}-{vpn_high:.3f}]",
            "Brier_int":  f"{brier_m:.3f} ± {brier_sd:.3f} [{brier_low:.3f}-{brier_high:.3f}]",
            "BSS_int":    f"{bss_int:.3f}"
        }

        # ── External validation ───────────────────────────────────────────────
        # Evaluate on two external sets:
        #   "FULL_rel"  -> all external samples (back-transformed miRNA values)
        #   "III_IV_rel" -> FIGO stages III-IV only
        for label, X_set, y_set in [
            ("FULL_rel",  X_rel_full, df_ex['ClassProg']),
            ("III_IV_rel", X_rel_filt, df_ex_filt['ClassProg'])
        ]:

            p_e = model_calib.predict_proba(X_set)[:, 1]
            y_e = (p_e >= t_oof).astype(int)

            # Bootstrap AUC CI for external set
            auc_sd_e, auc_low_e, auc_high_e = get_auc_bootstrap(y_set, p_e)

            # Exact binomial CI for recall on the external set
            n1_e = int(sum(y_set))
            tp_e = int(sum((y_set == 1) & (y_e == 1)))
            rec_low_e, rec_high_e = clopper_pearson(tp_e, n1_e)

            brier_ext = brier_score_loss(y_set, p_e)
            bss_ext   = 1 - (brier_ext / brier_baseline)

            # Calibration metrics for the external set
            c_int_e, c_slope_e, c_slope_sd_e, c_slope_ci_e = calibration_metrics_advanced(y_set, p_e)

            print(f"\n-- EXTERNAL VALIDATION - {label}")
            print("AUC:", roc_auc_score(y_set, p_e) if len(np.unique(y_set)) > 1 else np.nan)
            print("Recall:", recall_score(y_set, y_e))
            print("BalAcc:", balanced_accuracy_score(y_set, y_e))
            print("VPN:", vpn_score(y_set, y_e))
            print("Brier:", brier_ext)
            print("Brier Skill Score:", bss_ext)
            print("Calibration Intercept:", c_int_e)
            print("Calibration Slope:", c_slope_e)
            print("CM\n", confusion_matrix(y_set, y_e))

            # Append external validation columns to the result row
            row.update({
                f"AUC_ext_{label}":           roc_auc_score(y_set, p_e) if len(np.unique(y_set)) > 1 else np.nan,
                f"AUC_ext_{label}_SD":        auc_sd_e,
                f"AUC_ext_{label}_IC95":      f"[{auc_low_e:.3f}-{auc_high_e:.3f}]",
                f"Recall_ext_{label}":        recall_score(y_set, y_e),
                f"Recall_ext_{label}_IC95":   f"[{rec_low_e:.3f}-{rec_high_e:.3f}]",
                f"BalAcc_ext_{label}":        balanced_accuracy_score(y_set, y_e),
                f"VPN_ext_{label}":           vpn_score(y_set, y_e),
                f"Brier_ext_{label}":         brier_ext,
                f"BSS_ext_{label}":           bss_ext,
                f"CalIntercept_ext_{label}":  c_int_e,
                f"CalSlope_ext_{label}":      c_slope_e,
                f"CalSlope_ext_{label}_SD":   c_slope_sd_e,
                f"CalSlope_ext_{label}_IC95": c_slope_ci_e
            })

        results.append(row)


# =============================================================================
# EXPORT RESULTS & SAVE BEST MODEL
# =============================================================================

# Save full results table with all metrics, CIs, and calibration statistics
pd.DataFrame(results).to_csv(
    "results_final_RFE.csv",
    index=False
)

# Save the best model along with its decision threshold and selected features
joblib.dump({
    "model": best_global_model,
    "threshold": best_global_info["Threshold"],
    "features": best_global_info["Selected_features"]
}, "best_model_auc_v1.pkl")

# Also save the calibrated estimator on its own for convenience
joblib.dump(best_global_model, "best_model_auc_9mir.pkl")

print("\nBEST MODEL (AUC)")
print("Modelo:", best_global_info["Model"])
print("N_features:", best_global_info["N_features"])
print("Features:", best_global_info["Selected_features"])
print("Threshold:", best_global_info["Threshold"])
print("AUC:", best_global_auc)

