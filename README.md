[![DOI](https://zenodo.org/badge/1034709649.svg)](https://doi.org/10.5281/zenodo.17360240)



# miRNA-based Machine Learning Model for Risk Stratification in High-Grade Serous Ovarian Cancer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![R](https://img.shields.io/badge/R-4.0%2B-276DC3)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

> **Cristiane Esteves, Ms.C** — Bioinformatics and Computational Biology Laboratory (LBBC)  
> Brazilian National Cancer Institute (INCA-RJ) | [LBBC team](https://sites.google.com/view/bioinformaticainca-en/home-en)  
> ✉️ cristiane.esteves@ensino.inca.gov.br

---

## Overview

This repository contains the full analysis pipeline for the study:

**"miRNA-based machine learning model stratifies risk in high-grade serous ovarian cancer: a retrospective cohort study"**

Ovarian cancer, particularly the High-Grade Serous Ovarian Carcinoma (HGSOC) subtype, is the most lethal gynecological malignancy, mainly due to late-stage diagnosis and limited prognostic biomarkers. This study identifies and validates a prognostic miRNA signature combined with clinical variables using interpretable machine learning.

The trained model integrates **10 miRNA biomarkers** and **3 clinical variables** to classify patients as **Good** or **Poor** prognosis, with survival analysis supporting its independent prognostic value.

---

## Workflow

![Workflow](figure1.png)

---

## Repository Structure

```
prognosis_Predictor_miRNA/
│
├── data/                        # Input data files (see Data section below)
│   ├── top10_tcga.csv              # data for training
│   
│
├── scripts/                     # Python scripts
│   ├── ml_pipeline_miRNA_clinical.py    # Full ML training pipeline (RFE + GridSearch)
│   └── model_evaluation_external.py     # External validation + metrics + figures
│
├── requirements.txt             # Python dependencies
├── figure1.png                  # Workflow diagram
└── README.md
```

---

## Data

### Training cohort — TCGA-OV (public)

Training data derives from **The Cancer Genome Atlas Ovarian Cancer (TCGA-OV)** cohort, publicly available via the GSE164767.

- **Cohort:** HGSOC patients with miRNA expression profiling and clinical data
- **Target:** Overall Survival prognosis (Poor (<3 years) vs Good (>= 3 years))
- **miRNA quantification:** CPM (> 15 RPM)
- **Clinical variables:** age at diagnosis, FIGO stage, CA-125 (MUC16)


---



```
Raw input features
    └── 1. Scaler      → RobustScaler or StandardScaler (or passthrough)
    └── 2. Sampler     → SMOTE / RandomOverSampler / RandomUnderSampler (or passthrough)
    └── 3. RFE         → Recursive Feature Elimination (LogisticRegression base, C=0.01)
    └── 4. Classifier  → Best model selected by GridSearchCV
    └── 5. Calibration → Platt scaling (sigmoid), 3-fold CV
```

## Required Input Features

### miRNA features — 10 biomarkers

Column names must **contain** the following substrings. Values must be in **CPM** scale:

| Pattern | Example column name |
|---|---|
| `187` | `hsa_mir_187` |
| `149` | `hsa_mir_149` |
| `205` | `hsa_mir_205` |
| `150` | `hsa_mir_150` |
| `143` | `hsa_mir_143` |
| `485` | `hsa_mir_485` |
| `200a` | `hsa_mir_200a` |
| `23a` | `hsa_mir_23a` |
| `106b` | `hsa_mir_106b` |
| `151a` | `hsa_mir_151a` |

### Clinical features — 3 variables

| Column | Type | Preprocessing required |
|---|---|---|
| `age_at_index` | float | None |
| `figo_stage` | int | None (values 1–4) |
| `MUC16` | float | **Log2-transform:** `log2(raw_CA125 + 1)` |

### Target variable encoding

| Label | Value |
|---|---|
| Poor prognosis | `1` |
| Good prognosis | `0` |

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bioinformatics-inca/prognosis_Predictor_miRNA.git
cd prognosis_Predictor_miRNA
```

### 2. Set up Python environment

```bash
conda create -n mirna_prognosis python=3.9
conda activate mirna_prognosis
pip install -r requirements.txt

---

## Reproducing the Analysis

Run in the following order:

### Step 1 — Train the model (Python)
```bash
python scripts/ml_pipeline_miRNA_clinical.py
```

**Outputs:**
- `model/best_model_auc_v1.pkl` — best model (by internal AUC)
- `results_final_RFE.csv` — full results table with all metrics and CIs

### Step 2 — External validation (Python)

```bash
python scripts/model_evaluation_external.py
```

**Outputs:**
- `cm_rf_<cohort>.png` — annotated confusion matrix (300 dpi)
- `roc_curve_<cohort>.png` — ROC curve with AUC (300 dpi)
- Full metrics with 95% CI printed to console

---

**Key methodological decisions:**
- Threshold tuned on out-of-fold probabilities to prevent data leakage
- Probability calibration: Platt scaling (sigmoid), 3-fold internal CV
- Feature selection: RFE with LogisticRegression (C=0.01), sweeping 3–13 features
---

## Citation

If you use this code or model in your research, please cite:
The Lancet Health Americas [DOI}

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)**.  
You are free to share and adapt the material for any purpose, provided appropriate credit is given.  
🔗 [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)

---

## Contact

**Cristiane Esteves, M.sC**  
Bioinformatics and Computational Biology Laboratory (LBBC-INCA)  
Brazilian National Cancer Institute — Rio de Janeiro, Brazil  
✉️ cristiane.esteves@ensino.inca.gov.br  
🌐 [LBBC-INCA](https://sites.google.com/view/bioinformaticainca-en/home-en)

