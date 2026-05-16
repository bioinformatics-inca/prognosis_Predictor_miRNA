# ==============================================================================
# TCGA-OV miRNA Isoform Processing Pipeline
# ==============================================================================
# This script:
# 1. Loads normalized miRNA isoform expression data
# 2. Filters canonical isoforms
# 3. Removes recurrent/non-primary tumor samples
# 4. Integrates survival data from cBioPortal
# 5. Creates prognostic groups for ML analyses
#
# Prognostic definition:
# - Good: survival >= 3 years
# - Poor: death before 3 years
# - Patients censored before 3 years are excluded
#
# References:
# https://academic.oup.com/narcancer/article/3/1/zcab007/6168271
# https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164767
# ==============================================================================

library(data.table)
library(dplyr)

# ==============================================================================
# LOAD miRNA ISOFORM DATA
# ==============================================================================

miRNAisoform_norm <- as.data.frame(
  data.table::fread("data/TCGA-OV_corrected_tumor_median15.txt")
)

# Set isomiR names as rownames
rownames(miRNAisoform_norm) <- miRNAisoform_norm$isomiR

# Replace "|" by "." for R compatibility
miRNAisoform_norm$isomiR <- gsub("\\|", ".", miRNAisoform_norm$isomiR)

# Keep only canonical isoforms (.0.0.)
miRNAisoform_norm <- dplyr::filter(
  miRNAisoform_norm,
  grepl("\\.0.0.", miRNAisoform_norm$isomiR)
)

rownames(miRNAisoform_norm) <- miRNAisoform_norm$isomiR

# ==============================================================================
# PREPARE EXPRESSION MATRIX
# ==============================================================================

# Remove annotation columns and keep only expression matrix
miRNAisoform_norm <- miRNAisoform_norm[5:ncol(miRNAisoform_norm)]

# Transpose matrix: rows = samples | columns = miRNAs
miRNAisoform_norm <- as.data.frame(t(miRNAisoform_norm))

# Remove metastatic/recurrent samples ("-02A")
miRNAisoform_norm <- miRNAisoform_norm[
  !grepl("-02A", rownames(miRNAisoform_norm)),
]

# Temporary sample column
miRNAisoform_norm$sample <- rownames(miRNAisoform_norm)

# Remove known recurrent sample manually
miRNAisoform_norm <- miRNAisoform_norm[
  which(!(miRNAisoform_norm$sample == "TCGA-23-1023-01R-01R-1564-13")),
]

# Convert sample barcode to patient barcode
rownames(miRNAisoform_norm) <- substr(rownames(miRNAisoform_norm), 1, 12)

# Remove temporary column
miRNAisoform_norm$sample <- NULL

# ==============================================================================
# LOAD SURVIVAL DATA
# ==============================================================================

survival <- as.data.frame(
  data.table::fread("/data/Overall_Survival_(months).txt")
)

# Keep only patients present in expression matrix
survival <- survival[survival$`Patient ID` %in% rownames(miRNAisoform_norm),]

miRNAisoform_norm <- miRNAisoform_norm[
  rownames(miRNAisoform_norm) %in% survival$`Patient ID`,
]

# ==============================================================================
# CREATE SURVIVAL VARIABLES
# ==============================================================================

# Convert OS from months to years
survival$OS_years <- survival$OS_MONTHS / 12

# Truncate follow-up at 5 years
survival$time5 <- ifelse(survival$OS_years > 5, 5, survival$OS_years)

# Patients alive after 5 years are considered censored
survival$status5 <- ifelse(
  survival$OS_years > 5,
  "0:LIVING",
  survival$OS_STATUS
)

# ==============================================================================
# CREATE PROGNOSTIC GROUPS
# ==============================================================================

# Good prognosis: survival >= 3 years
survival$ClassProg2 <- ifelse(survival$time5 >= 3, "Good", NA)

# Poor prognosis: death before 3 years
survival$ClassProg2 <- ifelse(
  survival$time5 < 3 & survival$status5 == "1:DECEASED",
  "Poor",
  survival$ClassProg2
)

# Remove censored patients before 3 years
survival_2 <- survival[!is.na(survival$ClassProg2),]

# ==============================================================================
# MATCH EXPRESSION AND CLINICAL DATA
# ==============================================================================

miRNAisoform_norm <- miRNAisoform_norm[
  rownames(miRNAisoform_norm) %in% survival_2$`Patient ID`,
]

# Confirm matching order
all(survival_2$`Patient ID` == rownames(miRNAisoform_norm))
# TRUE

# ==============================================================================
# ADD CLINICAL LABELS
# ==============================================================================

# Add prognostic class
miRNAisoform_norm$Class <- survival_2$ClassProg2

# Remove NA labels if present
miRNAisoform_norm <- miRNAisoform_norm[!is.na(miRNAisoform_norm$Class),]

# Convert survival status to binary
# 1 = deceased | 0 = alive/censored
survival_2$status5 <- ifelse(
  survival_2$status5 == "1:DECEASED",
  1,
  0
)

# Clean canonical isoform suffix from column names
colnames(miRNAisoform_norm) <- gsub("\\.0.0.","",colnames(miRNAisoform_norm))

# Add survival variables
miRNAisoform_norm$time <- survival_2$time5
miRNAisoform_norm$status <- survival_2$status5

# ==============================================================================
# FINAL DATASET
# ==============================================================================

# Final object:
# miRNAisoform_norm
#
# Contains:
# - miRNA isoform expression
# - Prognostic class
# - Survival time
# - Survival status
# ==============================================================================
