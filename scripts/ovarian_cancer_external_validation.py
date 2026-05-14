#!/usr/bin/env python
# coding: utf-8

# # Prognostic Prediction in Ovarian Cancer
# 
# This notebook presents the external validation workflow for a prognostic machine learning model using circulating miRNAs and clinical variables in ovarian cancer.
# 
# ## Prognostic Stratification
# 
# - **Poor prognosis**: Death within 3 years after diagnosis and deceased status at follow-up
# - **Good prognosis**: survival beyond 3 years or alive at last follow-up
# 
# ## Features
# - miRNAs
# - Age at diagnosis
# - FIGO stage
# - MUC16 (CA125)
# 
# ## Model
# - Recursive Feature Elimination (RFE)
# - TabPFN classifier
# - Probability calibration using Platt scaling
# 
# ## Validation Cohorts
# - Adjuvant
# - NACT
# - Stage III-IV + Adjuvant
# 

# # =============================================================================
# # LOAD EXTERNAL VALIDATION DATAFRAMES
# # =============================================================================
# 
# Each CSV must contain the complete feature set used during model training,
# including all miRNA and clinical variables, as well as the target column:
# 
# - `ClassProg`
#     - Poor → 1
#     - Good → 0
# 
# ---
# 
# # REQUIRED miRNA FEATURES
# 
# The external datasets must contain the following circulating miRNA columns:
# 
# - `hsa-mir-143-3p`
# - `hsa-mir-149-5p`
# - `hsa-mir-150-5p`
# - `hsa-mir-151a-3p`
# - `hsa-mir-187-3p`
# - `hsa-mir-200a-5p`
# - `hsa-mir-205-5p`
# - `hsa-mir-23a-3p`
# - `hsa-mir-485-3p`
# - `hsa-mir-106b-3p`
# 
# miRNA values must be provided in **−ΔCt scale**.
# 
# The script automatically converts them into relative expression values using:
# 
# ```python
# 2 ** (-ΔCt)
# ```
# 
# during the preprocessing step.
# 
# ---
# 
# # REQUIRED CLINICAL FEATURES
# 
# - `age_at_index`
# - `figo_stage`
# - `MUC16`
# 
# ⚠️ IMPORTANT:
# 
# MUC16 (CA125) values must already be log2-transformed before inference:
# 
# ```python
# MUC16 = log2(MUC16_raw + 1)
# ```
# 
# ---
# 
# # EXTERNAL COHORTS
# 
# | Cohort | File | Description |
# |---|---|---|
# | Stage III-IV + Adjuvant | `df_ex_filt_adj.csv` | FIGO III-IV with adjuvant chemotherapy |
# | NACT | `df_ex_nact.csv` | Neoadjuvant chemotherapy cohort |
# | Adjuvant | `df_ex_adj.csv` | Adjuvant chemotherapy cohort |
# 

# In[1]:


import os
import pandas as pd
import numpy as np
import warnings
import scipy.stats as stats
from sklearn.utils import resample

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.metrics import (roc_auc_score, balanced_accuracy_score, accuracy_score, precision_score,
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

# =============================================================================
# LOAD EXTERNAL VALIDATION DATAFRAMES
# =============================================================================
# Each CSV must contain the full feature set (miRNA + clinical columns) and
# the target column 'ClassProg' (Poor=1, Good=0).
# MUC16 must already be log2-transformed; miRNA columns in −ΔCt scale
# (back-transformation to 2^(−ΔCt) is applied in the next section).

import pandas as pd

df_ex_adj = pd.read_csv(
    "data/df_ex_adj.csv",
    index_col=0
)

df_ex_nact = pd.read_csv(
    "data/df_ex_nact.csv",
    index_col=0
)

df_ex_filt_adj = pd.read_csv(
    "data/df_ex_filt_adj.csv",
    index_col=0
)

print("DataFrames loaded successfully.")


# In[2]:


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calcular_estatisticas(metric_value, n):
    """
    Compute analytical standard error (SE) and 95 % Wald CI for a proportion.

    Suitable for quick estimates; for small samples or extreme proportions,
    prefer Wilson interval (see wilson_confidence_interval below).

    Parameters
    ----------
    metric_value : float
        Observed proportion (e.g., accuracy, recall).
    n : int
        Sample size.

    Returns
    -------
    se, low, high : float
        Standard error and lower/upper 95 % CI bounds (clipped to [0, 1]).
    """

    if n <= 0:
        return 0.0, 0.0, 0.0

    p = max(0, min(1, metric_value))

    # Standard error of a proportion
    se = np.sqrt((p * (1 - p)) / n)

    low  = max(0, p - 1.96 * se)
    high = min(1, p + 1.96 * se)

    return se, low, high


def vpn_score(y_true, y_pred):
    """
    Compute the Negative Predictive Value (NPV).

    NPV = TN / (TN + FN)
    Proportion of negative predictions that are truly negative.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_pred : array-like
        Hard class predictions.

    Returns
    -------
    float
        NPV score, or 0.0 if no negative predictions were made.
    """

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    preditos_negativos = tn + fn

    if preditos_negativos > 0:
        return tn / preditos_negativos
    else:
        return 0.0


def calculate_ece(y_true, y_prob, n_bins=5):
    """
    Expected Calibration Error (ECE).

    Measures the average gap between predicted confidence and actual accuracy
    across equal-width probability bins.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    n_bins : int
        Number of calibration bins (default 5).

    Returns
    -------
    float
        ECE value (lower is better; 0 = perfect calibration).
    """

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= bin_lower) & (y_prob < bin_upper)
        if np.sum(mask) > 0:
            bin_conf = np.mean(y_prob[mask])
            bin_acc  = np.mean(y_true[mask])
            ece += np.abs(bin_conf - bin_acc) * np.sum(mask)

    return ece / len(y_true)


# =============================================================================
# LOAD SAVED MODEL
# =============================================================================
# The .pkl contains a dictionary with keys: "model", "threshold", "features".
# See the header docstring for full details of the internal pipeline steps.

import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

# Load the serialized model dictionary
model_9miR = joblib.load("best_model_auc_v1.pkl")


# =============================================================================
# CONFUSION MATRIX VISUALIZATION FUNCTION
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    """
    Render a styled confusion matrix as a Seaborn heatmap with per-cell
    counts, percentages, and a summary statistics footer.

    For binary matrices the footer reports:
      Recall (Sensitivity), Specificity, Balanced Accuracy, and NPV.

    Parameters
    ----------
    cf : ndarray of shape (n_classes, n_classes)
        Confusion matrix from sklearn.metrics.confusion_matrix.
    group_names : list[str], optional
        Custom label for each cell (e.g. ['TN','FP','FN','TP']).
    categories : list[str] or 'auto'
        Tick labels for axes.
    count : bool
        Show absolute counts inside each cell.
    percent : bool
        Show percentages (count / total) inside each cell.
    cbar : bool
        Display the colour bar.
    xyticks : bool
        Display axis tick labels.
    xyplotlabels : bool
        Display axis titles and summary stats.
    sum_stats : bool
        Append summary metrics to the x-axis label.
    figsize : tuple, optional
        Figure size (width, height) in inches.
    cmap : str
        Matplotlib/Seaborn colormap name.
    title : str, optional
        Plot title.
    """

    # ── Build per-cell annotation strings ────────────────────────────────────
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names) == cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten() / np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels, group_counts, group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0], cf.shape[1])

    # ── Build footer summary statistics ──────────────────────────────────────
    if sum_stats:
        acuracia = np.trace(cf) / float(np.sum(cf))

        if len(cf) == 2:
            # cf[1,1]=TP, cf[1,0]=FN, cf[0,1]=FP, cf[0,0]=TN
            recall          = cf[1, 1] / sum(cf[1, :])
            especificidade  = cf[0, 0] / sum(cf[0, :])
            balanced_acc    = (recall + especificidade) / 2
            # NPV = TN / (TN + FN)
            vpn = cf[0, 0] / (cf[0, 0] + cf[1, 0]) if (cf[0, 0] + cf[1, 0]) > 0 else 0

            stats_text = (
                "\n\nRecall (Sens.)={:0.3f}"
                "\nSpecificity={:0.3f}"
                "\nBalanced Acc.={:0.3f}"
                "\nNPV (VPN)={:0.3f}"
            ).format(recall, especificidade, balanced_acc, vpn)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(acuracia)
    else:
        stats_text = ""

    # ── Plot ─────────────────────────────────────────────────────────────────
    if figsize is None:
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks is False:
        categories = False

    plt.figure(figsize=figsize)
    sns.heatmap(cf, annot=box_labels, fmt="", cmap=cmap, cbar=cbar,
                xticklabels=categories, yticklabels=categories,
                annot_kws={"size": 12})

    if xyplotlabels:
        plt.ylabel('Observed prognosis')
        plt.xlabel('Predicted prognostic model' + stats_text)
    else:
        plt.xlabel(stats_text)

    if title:
        plt.title(title)


# =============================================================================
# FEATURE DEFINITION & EXTERNAL COHORT PREPROCESSING
# =============================================================================

model = joblib.load("best_model_auc_v1.pkl")


# =============================================================================
# EXTERNAL DATASET PREPARATION — SCALE CORRECTION FOR miRNA
# =============================================================================
# Training data used 2^(−ΔCt) relative expression.
# External cohort stores −ΔCt values → back-transform to 2^(−ΔCt).

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import scipy.stats as stats

from sklearn.metrics import (
    roc_auc_score, recall_score, balanced_accuracy_score,
    accuracy_score, confusion_matrix, brier_score_loss,
    roc_curve, auc
)
from sklearn.utils import resample
from statsmodels.stats.proportion import proportion_confint

# Identify miRNA feature columns by substring patterns in their names
mirna_patterns = ['187', '149', '205', '150', '143', '485', '200a', '23a', '106b', '151a']
mirna_cols = [c for c in df_ex_adj.columns if any(p in c for p in mirna_patterns)]
 
# Clinical covariates
clinical_cols = ['age_at_index', 'figo_stage', 'MUC16']
 
features = mirna_cols + clinical_cols

X_ex_adj   = df_ex_adj[features]
X_ex_nact  = df_ex_nact[features]
X_ex_filt_adj = df_ex_filt_adj[features]

# Identify miRNA columns (lowercase 'mir' substring convention)
cols_mir = [c for c in features if 'mir' in c.lower()]

# Back-transform all miRNA sets from −ΔCt to relative expression
X_rel_adj      = X_ex_adj.copy();      X_rel_adj[cols_mir]      = 2**(-X_ex_adj[cols_mir])
X_rel_nact     = X_ex_nact.copy();     X_rel_nact[cols_mir]     = 2**(-X_ex_nact[cols_mir])
X_rel_filt_adj = X_ex_filt_adj.copy(); X_rel_filt_adj[cols_mir] = 2**(-X_ex_filt_adj[cols_mir])


# =============================================================================
# ROBUST STATISTICAL FUNCTIONS
# =============================================================================

def wilson_confidence_interval(count, nobs):
    """
    Wilson score 95 % confidence interval for a binomial proportion.

    Preferred over Wald approximation for small samples or proportions
    near 0 or 1 (e.g., sensitivity/specificity in small cohorts).

    Parameters
    ----------
    count : int
        Number of successes (e.g., true positives for recall).
    nobs : int
        Total number of observations (e.g., all positives for recall).

    Returns
    -------
    low, high : float
        Lower and upper 95 % CI bounds.
    """

    if nobs == 0:
        return 0.0, 0.0

    low, high = proportion_confint(
        count, nobs, alpha=0.05, method='wilson'
    )

    return low, high


def delong_roc_variance(y_true, y_scores):
    """
    Estimate AUC standard error using the analytical DeLong method.

    More efficient than bootstrap for AUC CIs; avoids resampling variance.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_scores : array-like
        Predicted probabilities for the positive class.

    Returns
    -------
    float
        Standard error of the AUC estimate (use 1.96 × SE for 95 % CI).
        Returns np.nan if one class is absent.
    """

    y_true   = np.array(y_true)
    y_scores = np.array(y_scores)

    pos = y_true == 1
    neg = y_true == 0

    n_pos = np.sum(pos)
    n_neg = np.sum(neg)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    scores_pos = y_scores[pos]
    scores_neg = y_scores[neg]

    # V10: placement values for positive samples
    v10 = np.array([
        (np.sum(scores_neg < p) + 0.5 * np.sum(scores_neg == p)) / n_neg
        for p in scores_pos
    ])

    # V01: placement values for negative samples
    v01 = np.array([
        (np.sum(scores_pos > n) + 0.5 * np.sum(scores_pos == n)) / n_pos
        for n in scores_neg
    ])

    var = (np.var(v10, ddof=1) / n_pos) + (np.var(v01, ddof=1) / n_neg)

    return np.sqrt(var)


def calculate_ece(y_true, y_prob, n_bins=5):
    """
    Expected Calibration Error (ECE) — reused here with equal-width bins.

    See module-level docstring for method details.
    """

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0

    for i in range(n_bins):
        mask = (y_prob > bin_boundaries[i]) & (y_prob <= bin_boundaries[i + 1])
        if np.any(mask):
            ece += (
                np.abs(np.mean(y_prob[mask]) - np.mean(y_true[mask]))
                * np.mean(mask)
            )

    return ece


def get_bootstrap_metrics(y_true, y_prob, y_pred, n_iterations=1000, seed=42):
    """
    Compute 95 % bootstrap percentile CIs for balanced accuracy, accuracy,
    Brier score, and ECE using stratified resampling.

    Parameters
    ----------
    y_true : array-like
        Ground-truth binary labels.
    y_prob : array-like
        Predicted probabilities for the positive class.
    y_pred : array-like
        Hard class predictions (after threshold).
    n_iterations : int
        Number of bootstrap resamples (default 1 000).
    seed : int
        Base random seed (incremented per iteration for independence).

    Returns
    -------
    dict
        Keys: 'bal_acc', 'accuracy', 'brier', 'ece'.
        Values: tuple (ci_low, ci_high).
    """

    boot_stats = {'bal_acc': [], 'accuracy': [], 'brier': [], 'ece': []}

    for i in range(n_iterations):

        y_t_res, p_e_res, y_e_res = resample(
            y_true, y_prob, y_pred,
            stratify=y_true,
            random_state=seed + i
        )

        # Skip samples with a single class (metrics undefined)
        if len(np.unique(y_t_res)) < 2:
            continue

        boot_stats['bal_acc'].append(balanced_accuracy_score(y_t_res, y_e_res))
        boot_stats['accuracy'].append(accuracy_score(y_t_res, y_e_res))
        boot_stats['brier'].append(brier_score_loss(y_t_res, p_e_res))
        boot_stats['ece'].append(calculate_ece(y_t_res, p_e_res))

    return {
        k: (np.percentile(v, 2.5), np.percentile(v, 97.5))
        for k, v in boot_stats.items()
    }


# =============================================================================
# CONFIGURATION — MODEL & COHORTS
# =============================================================================

seed_value = 42

# Load model dictionary: keys → "model", "threshold", "features"
model_obj = joblib.load("best_model_auc_v1.pkl")

# External cohorts to evaluate: (display label, feature matrix, target series)
datasets = [
    ("Stage III-IV + Adjuvant", X_rel_filt_adj, df_ex_filt_adj['ClassProg']),
    ("NACT",                    X_rel_nact,     df_ex_nact['ClassProg']),
    ("Adjuvant",                X_rel_adj,      df_ex_adj['ClassProg'])
]


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

for label, X_set, y_set in datasets:

    # ── Predictions ──────────────────────────────────────────────────────────
    # predict_proba runs the full internal pipeline (scaler → sampler → RFE → clf → calibration)
    p_e = model_obj['model'].predict_proba(X_set)[:, 1]

    # Apply the pre-tuned threshold (maximises balanced accuracy at recall >= 83 %)
    y_e = (p_e >= model_obj['threshold']).astype(int)

    y_set_np = y_set.values

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_set_np, y_e).ravel()

    # ── AUC (DeLong CI) ──────────────────────────────────────────────────────
    auc_p  = roc_auc_score(y_set_np, p_e)
    auc_se = delong_roc_variance(y_set_np, p_e)
    auc_low  = max(0, auc_p - 1.96 * auc_se)
    auc_high = min(1, auc_p + 1.96 * auc_se)

    # ── Balanced Accuracy & Accuracy (bootstrap CI) ───────────────────────
    bal_acc = balanced_accuracy_score(y_set_np, y_e)
    acc_p   = accuracy_score(y_set_np, y_e)

    # ── Recall / Sensitivity (Wilson CI) ─────────────────────────────────
    rec_p = tp / (tp + fn) if (tp + fn) > 0 else 0
    rec_low, rec_high = wilson_confidence_interval(tp, tp + fn)

    # ── Specificity (Wilson CI) ───────────────────────────────────────────
    spec_p = tn / (tn + fp) if (tn + fp) > 0 else 0
    spec_low, spec_high = wilson_confidence_interval(tn, tn + fp)

    # ── NPV (Wilson CI) ───────────────────────────────────────────────────
    npv_p = tn / (tn + fn) if (tn + fn) > 0 else 0
    npv_low, npv_high = wilson_confidence_interval(tn, tn + fn)

    # ── Brier Score & ECE ─────────────────────────────────────────────────
    brier_p = brier_score_loss(y_set_np, p_e)
    ece_p   = calculate_ece(y_set_np, p_e)

    # ── Bootstrap CIs for remaining metrics ──────────────────────────────
    ic_boot = get_bootstrap_metrics(
        y_set_np, p_e, y_e, n_iterations=1000, seed=seed_value
    )

    # ── Print report ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RESULTS: {label}")
    print(f"{'='*70}")
    print(f"AUC:               {auc_p:.4f} | 95% CI: [{auc_low:.4f} - {auc_high:.4f}]")
    print(f"Balanced Accuracy: {bal_acc:.4f} | 95% CI: [{ic_boot['bal_acc'][0]:.4f} - {ic_boot['bal_acc'][1]:.4f}]")
    print(f"Accuracy:          {acc_p:.4f} | 95% CI: [{ic_boot['accuracy'][0]:.4f} - {ic_boot['accuracy'][1]:.4f}]")
    print(f"Recall:            {rec_p:.4f} | 95% CI: [{rec_low:.4f} - {rec_high:.4f}]")
    print(f"Specificity:       {spec_p:.4f} | 95% CI: [{spec_low:.4f} - {spec_high:.4f}]")
    print(f"NPV:               {npv_p:.4f} | 95% CI: [{npv_low:.4f} - {npv_high:.4f}]")
    print(f"Brier Score:       {brier_p:.4f} | 95% CI: [{ic_boot['brier'][0]:.4f} - {ic_boot['brier'][1]:.4f}]")
    print(f"ECE:               {ece_p:.4f} | 95% CI: [{ic_boot['ece'][0]:.4f} - {ic_boot['ece'][1]:.4f}]")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_set_np, y_e)}")

    # ── Confusion matrix figure ───────────────────────────────────────────
    plt.figure(figsize=(6, 5))

    labels_cm  = ['True Negative', 'False Positive', 'False Negative', 'True Positive']
    categories = ['Good', 'Poor']

    make_confusion_matrix(
        confusion_matrix(y_set_np, y_e),
        group_names=labels_cm,
        categories=categories,
        cmap='binary',
        title=f"INCA-OV {label}"
    )

    cm_name = f"cm_rf_{label}.png"
    plt.savefig(cm_name, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {cm_name}")

    # ── ROC curve figure ──────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_set_np, p_e)
    roc_auc_val = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='black', lw=2, label=f'ROC curve (AUC = {roc_auc_val:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')   # Random classifier baseline
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title(f'Receiver Operating Characteristic - {label}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    roc_name = f"roc_curve_{label}.png"
    plt.savefig(roc_name, format="png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved: {roc_name}")

