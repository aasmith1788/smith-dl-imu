library(ggplot2)
library(gridExtra)
library(grid)

# Naive Baseline Results Data
data <- data.frame(
  Axis = c("X", "Y", "Z"),
  Constant_Baseline = c(1.3338, -1.6536, 0.2636),
  Training_Range = c("16.56° (-4.18° to 12.37°)", "10.21° (-8.09° to 2.12°)", "5.21° (-1.39° to 3.82°)"),
  Global_RMSE = c("2.426°", "1.376°", "0.727°"),
  Global_nRMSE = c("14.65%", "13.47%", "13.94%"),
  Global_Corr = c("0.000", "0.000", "0.000"),
  PerTrial_nRMSE = c("14.14% ± 3.84%", "12.96% ± 3.70%", "13.32% ± 4.13%"),
  PerTrial_Corr = c("0.000 ± 0.000", "0.000 ± 0.000", "0.000 ± 0.000"),
  N_Trials = c(876, 876, 876)
)

# Build table as grob
table_grob <- tableGrob(
  data.frame(
    "Axis"                   = data$Axis,
    "Constant\nBaseline"     = data$Constant_Baseline,
    "Training Range"         = data$Training_Range,
    "Global\nRMSE"           = data$Global_RMSE,
    "Global\nnRMSE"          = data$Global_nRMSE,
    "Global\nCorr"           = data$Global_Corr,
    "Per-Trial\nnRMSE"       = data$PerTrial_nRMSE,
    "Per-Trial\nCorr"        = data$PerTrial_Corr,
    "N Trials"               = data$N_Trials,
    check.names = FALSE
  ),
  rows = NULL,
  theme = ttheme_default(
    core = list(
      fg_params = list(fontsize = 11),
      bg_params = list(
        fill = c("#f7f7f7", "#ffffff", "#f7f7f7"),  # alternating row colors
        col  = "grey80"
      )
    ),
    colhead = list(
      fg_params = list(fontsize = 12, fontface = "bold", col = "white"),
      bg_params = list(fill = "#2c3e50", col = "grey80")
    )
  )
)

# Add title
title <- textGrob(
  "True Naive Baseline Results — Per Axis",
  gp = gpar(fontsize = 15, fontface = "bold")
)

subtitle <- textGrob(
  "Baseline predicts a constant flat line (overall training mean) for every timepoint.\nCorrelation is ~0 by definition — no temporal structure captured.",
  gp = gpar(fontsize = 10, col = "grey40")
)

# Combine and save
png("C:/Users/aasmi/smith-dl-imu/Notes/naive_baseline_table.png", width = 1200, height = 320, res = 120)
grid.arrange(
  title,
  subtitle,
  table_grob,
  heights = c(0.15, 0.12, 0.73)
)
dev.off()

message("Saved: C:/Users/aasmi/smith-dl-imu/Notes/naive_baseline_table.png")


# ==============================================================================
# PyramidAttnCNN v3 — Knee Joint Moment Results (Train & Test)
# ==============================================================================

train_data <- data.frame(
  Axis         = c("X", "Y", "Z", "AVG"),
  PerTrial_Corr = c("0.966 ± 0.044", "0.946 ± 0.052", "0.970 ± 0.033", "—"),
  Global_Corr  = c("0.956", "0.935", "0.958", "0.950"),
  Global_RMSE  = c("0.711°", "0.512°", "0.212°", "0.478°"),
  Global_nRMSE = c("4.29%", "5.02%", "4.06%", "4.46%"),
  PerTrial_nRMSE = c("3.99% ± 1.59%", "4.68% ± 1.81%", "3.80% ± 1.43%", "—"),
  SD_Ratio     = c("0.636", "0.682", "0.653", "0.657"),
  N_Trials     = c("3504/3504", "3504/3504", "3504/3504", "—")
)

test_data <- data.frame(
  Axis         = c("X", "Y", "Z", "AVG"),
  PerTrial_Corr = c("0.926 ± 0.116", "0.918 ± 0.072", "0.945 ± 0.064", "—"),
  Global_Corr  = c("0.855", "0.844", "0.887", "0.862"),
  Global_RMSE  = c("1.267°", "0.756°", "0.339°", "0.787°"),
  Global_nRMSE = c("7.65%", "7.40%", "6.50%", "7.18%"),
  PerTrial_nRMSE = c("6.84% ± 3.42%", "6.70% ± 3.15%", "5.98% ± 2.55%", "—"),
  SD_Ratio     = c("0.447", "0.499", "0.453", "0.466"),
  N_Trials     = c("876/876", "876/876", "876/876", "—")
)

make_model_table <- function(df, row_fill) {
  tableGrob(
    data.frame(
      "Axis"               = df$Axis,
      "Per-Trial\nCorr (μ±σ)"  = df$PerTrial_Corr,
      "Global\nCorr"       = df$Global_Corr,
      "Global\nRMSE"       = df$Global_RMSE,
      "Global\nnRMSE"      = df$Global_nRMSE,
      "Per-Trial\nnRMSE (μ±σ)" = df$PerTrial_nRMSE,
      "SD\nRatio"          = df$SD_Ratio,
      "N Trials"           = df$N_Trials,
      check.names = FALSE
    ),
    rows = NULL,
    theme = ttheme_default(
      core = list(
        fg_params = list(fontsize = 10),
        bg_params = list(
          fill = row_fill,
          col  = "grey80"
        )
      ),
      colhead = list(
        fg_params = list(fontsize = 11, fontface = "bold", col = "white"),
        bg_params = list(fill = "#2c3e50", col = "grey80")
      )
    )
  )
}

# Alternating fills: 4 rows (3 axes + AVG)
train_fill <- c("#eaf4fb", "#ffffff", "#eaf4fb", "#d0e8f5")
test_fill  <- c("#fef9e7", "#ffffff", "#fef9e7", "#fdebd0")

train_table <- make_model_table(train_data, train_fill)
test_table  <- make_model_table(test_data,  test_fill)

main_title <- textGrob(
  "PyramidAttnCNN v3 — Knee Joint Moment Prediction Results",
  gp = gpar(fontsize = 15, fontface = "bold")
)
main_subtitle <- textGrob(
  "3-axis (X, Y, Z) knee joint moments predicted from IMU data. Metrics normalized by per-axis training range.",
  gp = gpar(fontsize = 10, col = "grey40")
)

train_label <- textGrob("Training Set  (n = 3,504 trials)",
                        gp = gpar(fontsize = 12, fontface = "bold", col = "#1a6fa8"))
test_label  <- textGrob("Test Set  (n = 876 trials)",
                        gp = gpar(fontsize = 12, fontface = "bold", col = "#b7770d"))

png("C:/Users/aasmi/smith-dl-imu/Notes/v3_results_table.png",
    width = 1400, height = 560, res = 120)
grid.arrange(
  main_title,
  main_subtitle,
  train_label,
  train_table,
  test_label,
  test_table,
  heights = c(0.08, 0.06, 0.06, 0.32, 0.06, 0.32),
  padding = unit(0.5, "line")
)
dev.off()

message("Saved: C:/Users/aasmi/smith-dl-imu/Notes/v3_results_table.png")
