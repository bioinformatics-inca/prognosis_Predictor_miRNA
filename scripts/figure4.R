#!/usr/bin/env Rscript

# ==============================================================================
# Figure 4 - miRNA/mRNA integrated analysis
# ==============================================================================
# Description:
#   Reproducible script to generate Figure 4 panels:
#   1) RNA-seq heatmap of top DEGs
#   2) miRNA-target interaction network
#   3) Tumor exosome vs tumor cell miRNA expression
#   4) GSE106817 circulating miRNA expression across clinical groups
#   5) miRNA pathway enrichment dotplot
#
# Input:
#   - An .RData file containing the objects required by each panel.
#   - Optional GEO download for GSE106817.
#
# Expected objects in the .RData file:
#   counts_log10, resSig_top100, hc, ann, coll, cores_heatmap,
#   DEG_target, cells_long, label_fun, df
#
# Author: Cristiane Esteves and Helena Zancanaro
# ==============================================================================

# ----------------------------- 1. Configuration -------------------------------

config <- list(
  input_rdata = "data/fig4.RData",
  output_dir = "results/figure4",
  gse_id = "GSE106817",
  top_mirnas = c(
    "hsa-mir-106b-3p",
    "hsa-mir-143-3p",
    "hsa-mir-149-5p",
    "hsa-mir-150-5p",
    "hsa-mir-151a-3p",
    "hsa-mir-187-3p",
    "hsa-mir-200a-5p",
    "hsa-mir-205-5p",
    "hsa-mir-23a-3p",
    "hsa-mir-485-3p"
  )
)

dir.create(config$output_dir, recursive = TRUE, showWarnings = FALSE)

# ----------------------------- 2. Packages ------------------------------------

required_packages <- c(
  "Biobase",
  "dplyr",
  "GEOquery",
  "ggplot2",
  "ggplotify",
  "ggpubr",
  "ggrepel",
  "ggthemes",
  "ggraph",
  "igraph",
  "pheatmap",
  "readr",
  "scales",
  "tibble",
  "tidyr"
)

install_if_missing <- function(packages) {
  missing_packages <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_packages) > 0) {
    stop(
      "Missing packages: ",
      paste(missing_packages, collapse = ", "),
      "\nInstall them before running this script. For Bioconductor packages, use BiocManager::install().",
      call. = FALSE
    )
  }
}

install_if_missing(required_packages)

suppressPackageStartupMessages({
  library(Biobase)
  library(dplyr)
  library(GEOquery)
  library(ggplot2)
  library(ggplotify)
  library(ggpubr)
  library(ggrepel)
  library(ggthemes)
  library(ggraph)
  library(igraph)
  library(pheatmap)
  library(readr)
  library(scales)
  library(tibble)
  library(tidyr)
})

# ----------------------------- 3. Helpers -------------------------------------

check_required_objects <- function(objects) {
  missing_objects <- objects[!vapply(objects, exists, logical(1), envir = .GlobalEnv)]
  if (length(missing_objects) > 0) {
    stop(
      "The following required objects are missing from the loaded .RData file: ",
      paste(missing_objects, collapse = ", "),
      call. = FALSE
    )
  }
}

format_p_lancet <- function(p) {
  if (is.na(p)) {
    return(NA_character_)
  }

  if (p < 1e-4) {
    return(" < 0\u00b70001")
  }

  paste0(" = ", signif(p, 2))
}

theme_manuscript_safe <- function() {
  if (exists("theme_manuscript", mode = "function")) {
    return(theme_manuscript())
  }

  theme_bw(base_size = 14)
}

save_plot <- function(filename, plot, width, height, dpi = 300, bg = "white") {
  ggplot2::ggsave(
    filename = file.path(config$output_dir, filename),
    plot = plot,
    width = width,
    height = height,
    units = "in",
    dpi = dpi,
    bg = bg
  )
}

# ----------------------------- 4. Load data -----------------------------------

if (!file.exists(config$input_rdata)) {
  stop(
    "Input file not found: ", config$input_rdata,
    "\nUpdate config$input_rdata before running this script.",
    call. = FALSE
  )
}

load(config$input_rdata)

check_required_objects(c(
  "counts_log10",
  "resSig_top100",
  "hc",
  "ann",
  "coll",
  "cores_heatmap",
  "DEG_target",
  "cells_long",
  "label_fun",
  "df"
))

# ----------------------------- 5. Heatmap RNA-seq -----------------------------

plot_heatmap_rnaseq <- function() {
  heatmap_plot <- pheatmap(
    mat = as.matrix(counts_log10[rownames(resSig_top100), ][1:30, ]),
    scale = "row",
    name = "Expression",
    cluster_rows = FALSE,
    cluster_cols = hc,
    show_rownames = TRUE,
    show_colnames = FALSE,
    fontsize_row = 10,
    fontsize_col = 2,
    annotation_col = ann[1],
    annotation_colors = coll,
    color = cores_heatmap,
    main = "padj < 0.01 | Pearson corr. (TCGA-OV tumor vs GTEx normal) - top 30"
  ) %>%
    ggplotify::as.ggplot()

  save_plot(
    filename = "heatmap_RNAseq.svg",
    plot = heatmap_plot,
    width = 8.5,
    height = 12
  )

  invisible(heatmap_plot)
}

# ----------------------------- 6. miRNA-target network ------------------------

plot_mirna_target_network <- function(top_mirnas = config$top_mirnas) {
  network_data <- DEG_target %>%
    filter(tolower(miRNA) %in% top_mirnas)

  edges <- network_data %>%
    select(from = miRNA, to = Target)

  mirna_nodes <- network_data %>%
    group_by(miRNA) %>%
    summarise(logFC = mean(log2FoldChange, na.rm = TRUE), .groups = "drop") %>%
    rename(name = miRNA) %>%
    mutate(type = "miRNA")

  target_nodes <- network_data %>%
    group_by(Target) %>%
    summarise(logFC = mean(log2FoldChange, na.rm = TRUE), .groups = "drop") %>%
    rename(name = Target) %>%
    mutate(type = "target")

  nodes <- bind_rows(mirna_nodes, target_nodes) %>%
    distinct(name, .keep_all = TRUE)

  graph <- graph_from_data_frame(d = edges, vertices = nodes, directed = FALSE)

  V(graph)$type <- nodes$type
  V(graph)$logFC <- nodes$logFC
  V(graph)$name <- nodes$name

  set.seed(123)
  graph_layout <- create_layout(graph, layout = "stress")

  mirna_labels <- graph_layout %>%
    filter(type == "miRNA")

  target_labels <- graph_layout %>%
    filter(type != "miRNA", abs(logFC) > 5.1)

  network_plot <- ggraph(graph_layout) +
    geom_edge_link(alpha = 0.2, color = "gray40") +
    geom_node_point(aes(size = abs(logFC), color = logFC)) +
    geom_label_repel(
      data = mirna_labels,
      aes(x = x, y = y, label = name),
      size = 3.5,
      fill = "white",
      fontface = "bold",
      color = "black",
      box.padding = 0.5
    ) +
    geom_text_repel(
      data = target_labels,
      aes(x = x, y = y, label = name),
      size = 3,
      color = "black",
      fontface = "bold",
      box.padding = 0.2
    ) +
    scale_color_gradient2(
      low = "blue",
      mid = "white",
      high = "red",
      midpoint = 0,
      name = "log2FC"
    ) +
    scale_size_continuous(name = "abs(log2FC)") +
    theme_void() +
    theme(
      legend.position = "right",
      legend.box = "vertical"
    )

  network_plot <- ggplotify::as.ggplot(network_plot)

  save_plot(
    filename = "network_interaction.png",
    plot = network_plot,
    width = 15.3,
    height = 13.3
  )

  invisible(network_plot)
}

# ----------------------------- 7. Exosome enrichment --------------------------

plot_exosome_enrichment <- function(top_mirnas = config$top_mirnas) {
  exosome_plot <- ggplot(
    data = subset(
      cells_long,
      type %in% c("Tumor_exo", "Tumor_cell") & variable %in% top_mirnas
    ),
    aes(x = type, y = value, fill = type)
  ) +
    geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.8) +
    geom_jitter(width = 0.2, size = 0.5, alpha = 0.5, color = "black") +
    facet_wrap(
      ~ variable,
      scales = "free_y",
      labeller = labeller(variable = label_fun),
      nrow = 1
    ) +
    scale_y_continuous(trans = "log10") +
    scale_fill_manual(values = c("Tumor_exo" = "#1f80b4", "Tumor_cell" = "#DB8090")) +
    labs(
      y = expression("miRNA expression (log"[10] * ")"),
      fill = "Cell type"
    ) +
    stat_compare_means(
      method = "wilcox.test",
      label.y.npc = "top",
      size = 3.5
    ) +
    theme_manuscript_safe() +
    theme(
      axis.text.x = element_blank(),
      axis.ticks.x = element_blank(),
      axis.title.x = element_blank(),
      axis.text.y = element_text(size = 12),
      axis.title.y = element_text(size = 16),
      legend.title = element_text(size = 14),
      legend.text = element_text(size = 12),
      strip.text = element_text(size = 12, color = "black", face = "bold"),
      panel.background = element_rect(fill = "white", color = NA),
      plot.background = element_rect(fill = "white", color = NA),
      strip.background = element_rect(fill = "white", color = NA),
      panel.grid = element_blank()
    )

  save_plot(
    filename = "cell_expression.png",
    plot = exosome_plot,
    width = 17,
    height = 6
  )

  invisible(exosome_plot)
}

# ----------------------------- 8. GSE106817 boxplots --------------------------

download_gse106817 <- function(gse_id = config$gse_id) {
  gse_list <- getGEO(
    gse_id,
    GSEMatrix = TRUE,
    getGPL = FALSE
  )

  gse <- gse_list[[1]]
  expr <- Biobase::exprs(gse) %>%
    as.data.frame()

  metadata <- pData(gse) %>%
    rownames_to_column("sample_id")

  gpl_id <- annotation(gse)
  gpl <- getGEO(gpl_id)
  gpl_table <- Table(gpl)

  list(
    expr = expr,
    metadata = metadata,
    gpl_table = gpl_table
  )
}

prepare_gse106817_metadata <- function(metadata) {
  other_cancers <- c(
    "Breast Cancer",
    "Colorectal Cancer",
    "Esophageal Cancer",
    "Gastric Cancer",
    "Hepatocellular Carcinoma",
    "Lung Cancer",
    "Pancreatic Cancer",
    "Sarcoma"
  )

  metadata %>%
    select(sample_id, description) %>%
    filter(description != "OV_others") %>%
    mutate(
      group = case_when(
        description == "non-Cancer" ~ "Non-cancer Controls",
        description == "Ovarian Cancer" ~ "Ovarian Cancer",
        description == "Borderline Ovarian Tumor" ~ "Borderline Ovarian Tumor",
        description == "Benign Ovarian Disease" ~ "Benign Ovarian Disease",
        description %in% other_cancers ~ "Other Solid Cancers",
        TRUE ~ NA_character_
      )
    )
}

plot_gse106817_boxplots <- function(top_mirnas = config$top_mirnas) {
  gse_data <- download_gse106817(config$gse_id)

  sample_ids <- prepare_gse106817_metadata(gse_data$metadata)

  filtered_gpl <- gse_data$gpl_table %>%
    filter(miRNA_ID_LIST %in% top_mirnas)

  mirna_codes <- filtered_gpl$miRNA

  filtered_matrix <- gse_data$expr[rownames(gse_data$expr) %in% mirna_codes, ]

  id_to_mirna <- filtered_gpl$miRNA_ID_LIST
  names(id_to_mirna) <- filtered_gpl$miRNA

  rownames(filtered_matrix) <- id_to_mirna[rownames(filtered_matrix)]
  rownames(filtered_matrix) <- gsub("miR", "mir", rownames(filtered_matrix))

  plot_dir <- file.path(config$output_dir, "GSE106817_boxplots")
  dir.create(plot_dir, recursive = TRUE, showWarnings = FALSE)

  group_colors <- c(
    "Non-cancer Controls" = "#4E79A7",
    "Ovarian Cancer" = "#E15759",
    "Borderline Ovarian Tumor" = "#F28E2B",
    "Benign Ovarian Disease" = "#59A14F",
    "Other Solid Cancers" = "#B07AA1"
  )

  wilcox_results <- list()

  for (mirna in rownames(filtered_matrix)) {
    mirna_file_name <- gsub("-", "_", mirna)

    box_df <- data.frame(
      sample_id = colnames(filtered_matrix),
      Expression = as.numeric(filtered_matrix[mirna, ])
    ) %>%
      left_join(sample_ids[, c("sample_id", "group")], by = "sample_id") %>%
      filter(!is.na(group), !is.na(Expression))

    shapiro_res <- box_df %>%
      group_by(group) %>%
      summarise(
        p_shapiro = ifelse(n() >= 3, shapiro.test(Expression)$p.value, NA_real_),
        .groups = "drop"
      )

    normal_distribution <- all(shapiro_res$p_shapiro > 0.05, na.rm = TRUE)

    if (length(unique(box_df$group)) <= 1) {
      next
    }

    if (normal_distribution) {
      global_res <- compare_means(Expression ~ group, data = box_df, method = "anova")
      global_test <- "ANOVA"
      pairwise_method <- "t.test"
    } else {
      global_res <- compare_means(Expression ~ group, data = box_df, method = "kruskal.test")
      global_test <- "Kruskal-Wallis"
      pairwise_method <- "wilcox.test"
    }

    p_global <- global_res$p
    p_label <- format_p_lancet(p_global)

    pairwise_comparisons <- list()

    if (!is.na(p_global) && p_global <= 0.05) {
      pairwise_res <- compare_means(
        Expression ~ group,
        data = box_df,
        method = pairwise_method
      )

      pairwise_sig <- pairwise_res %>%
        filter(p <= 0.05)

      pairwise_comparisons <- mapply(
        c,
        pairwise_sig$group1,
        pairwise_sig$group2,
        SIMPLIFY = FALSE
      )

      if (pairwise_method == "wilcox.test") {
        wilcox_results[[mirna_file_name]] <- pairwise_res[, c("group1", "group2", "p")]
      }
    }

    y_min <- min(box_df$Expression, na.rm = TRUE)
    y_max <- max(box_df$Expression, na.rm = TRUE)
    y_pad <- 0.1 * (y_max - y_min)

    if (length(pairwise_comparisons) > 0) {
      y_positions <- seq(
        from = y_max + y_pad,
        by = y_pad,
        length.out = length(pairwise_comparisons)
      )
    } else {
      y_positions <- NULL
    }

    box_plot <- ggplot(box_df, aes(x = group, y = Expression, fill = group)) +
      geom_boxplot(outlier.shape = NA, alpha = 0.7, color = "black", coef = Inf) +
      geom_jitter(width = 0.15, size = 0.5, alpha = 0.1) +
      scale_fill_manual(values = group_colors) +
      labs(
        title = mirna,
        subtitle = paste0("Global p (", global_test, "): ", p_label),
        y = expression("Expression log"[2] * " (microarray signal)"),
        x = "Group"
      ) +
      theme_bw(base_size = 16) +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold"),
        axis.text = element_text(color = "black"),
        panel.grid = element_blank(),
        legend.position = "right",
        legend.title = element_text(face = "bold"),
        legend.text = element_text(size = 16),
        axis.text.x = element_blank(),
        axis.ticks.x = element_blank(),
        axis.title.x = element_blank()
      )

    if (length(pairwise_comparisons) > 0) {
      box_plot <- box_plot +
        stat_compare_means(
          label = "p.signif",
          method = pairwise_method,
          comparisons = pairwise_comparisons,
          y.position = y_positions,
          hide.ns = TRUE
        )
    }

    ggsave(
      filename = file.path(plot_dir, paste0(mirna_file_name, "_groups_GSE106817.svg")),
      plot = box_plot,
      width = 9,
      height = 6,
      units = "in",
      dpi = 300
    )
  }

  if (length(wilcox_results) > 0) {
    wilcox_table <- bind_rows(wilcox_results, .id = "miRNA")
    readr::write_csv(wilcox_table, file.path(plot_dir, "pairwise_wilcox_results.csv"))
  }

  invisible(wilcox_results)
}

# ----------------------------- 9. Pathway enrichment dotplot ------------------

plot_pathway_dotplot <- function(top_mirnas = config$top_mirnas) {
  pathway_df <- df %>%
    filter(miRNA != "hsa-mir-191-5p")

  pathway_plot <- ggplot(pathway_df, aes(x = miRNA, y = Term_grouped, size = Score, color = pval)) +
    geom_point(alpha = 0.9) +
    scale_color_gradient(low = "red", high = "blue", name = "p-value") +
    scale_size(range = c(2, 5), name = "Enrichment Score") +
    theme_classic(base_size = 12, base_family = "Arial") +
    labs(x = "", y = "Pathways") +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1, size = 14, face = "bold"),
      axis.text.y = element_text(size = 14),
      axis.title = element_text(size = 14),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 12),
      panel.grid.major = element_line(color = "gray90"),
      panel.grid.minor = element_blank()
    )

  save_plot(
    filename = "pathways.svg",
    plot = pathway_plot,
    width = 9,
    height = 12
  )

  invisible(pathway_plot)
}

# ----------------------------- 10. Run analysis -------------------------------

message("Generating Figure 4 panels...")

plot_heatmap_rnaseq()
plot_mirna_target_network()
plot_exosome_enrichment()
plot_gse106817_boxplots()
plot_pathway_dotplot()

message("Done. Results saved to: ", normalizePath(config$output_dir))
