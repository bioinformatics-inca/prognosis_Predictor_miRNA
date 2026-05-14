[![DOI](https://zenodo.org/badge/1034709649.svg)](https://doi.org/10.5281/zenodo.17360240)
# miRNA-based machine learning model stratifies risk in high-grade serous ovarian cancer: a retrospective cohort study
Background: Ovarian cancer, particularly the high-grade serous ovarian carcinoma (HGSOC) subtype, is the most lethal gynecological malignancy, mainly due to late-stage diagnosis and limited prognostic biomarkers. Current clinical markers, such as CA125, have limited prognostic accuracy for risk stratification. MicroRNAs (miRNAs) have emerged as promising biomarkers due to their roles in tumor biology and stability in biofluids. This study aimed to identify and validate prognostic miRNA biomarkers in HGSOC.
Methods: A machine learning pipeline was implemented to develop a prognostic model using miRNA and clinical data. Candidate miRNAs were identified through feature selection and differential expression analyses. Recursive Feature Elimination (RFE) determined the optimal predictor set among miRNAs combined with age, stage, and MUC16. Model development used a hold-out split, with hyperparameter optimization under 5-fold cross-validation within the training set. Performance was evaluated using AUC, recall, and balanced accuracy. SHAP analysis assessed feature contributions, while enrichment analyses characterized miRNA–mRNA interactions and pathways.
Findings: The final model, incorporating 9 miRNAs with clinical variables, achieved an AUC of 0.762 [95%CI: 0.621–0.903], exceeding previously reported signatures. Key miRNAs, including hsa-miR-205-5p and hsa-miR-150-5p, were associated with angiogenesis, invasion, and chemoresistance pathways. In RT-qPCR validation, discriminative performance decreased; however, the continuous risk score remained independently associated with overall survival
Interpretation: We present an interpretable miRNA-based prognostic model for HGSOC integrating molecular and clinical features. Although ROC-based discrimination was limited in external validation, survival analyses supported independent prognostic value, with the continuous risk score significantly associated with overall survival. Continuous and classification-based stratification identified distinct survival groups, supporting the clinical relevance of the model and identified miRNA signature.

# Workflow
![This is an image](figure1.png)

---------------------------------------------------------------------------------------------------------
<sub><sup>
Bioinformatics and Computational Biology Laboratory (LBBC-INCA);
Brazilian National Cancer Institute (INCA-RJ) | LBBC team (https://sites.google.com/view/bioinformaticainca-en/home-en)
Developed by Cristiane Esteves, Ms.C
</sup></sub>

