library(ggplot2)
library(gridExtra)
library(grid)

# Naive Baseline Results Data
data <- data.frame(
  Axis = c("X", "Y", "Z"),
  Constant_Baseline = c(1.3338, -1.6536, 0.2636),
  Training_Range = c("16.56 (-4.18 to 12.37)", "10.21 (-8.09 to 2.12)", "5.21 (-1.39 to 3.82)"),
  Global_RMSE = c("2.426", "1.376", "0.727"),
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
    "Training Range\n(Nm/BW·Ht)" = data$Training_Range,
    "Global RMSE\n(Nm/BW·Ht)"  = data$Global_RMSE,
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
  "Baseline predicts a constant flat line (overall training mean) for every timepoint.\nCorrelation is ~0 by definition — no temporal structure captured.  |  Units: Nm/BW·Ht",
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
# Naive Baseline — Angles
# ==============================================================================

angle_naive_data <- data.frame(
  Axis               = c("X", "Y", "Z"),
  Constant_Baseline  = c(-20.4032, 0.2317, -5.6495),
  Training_Range     = c("75.98° (-67.41° to 8.56°)", "27.80° (-13.18° to 14.61°)", "55.54° (-41.26° to 14.28°)"),
  Global_RMSE        = c("10.33°", "4.90°", "6.11°"),
  Global_nRMSE       = c("13.60%", "17.63%", "11.00%"),
  Global_Corr        = c("0.000", "0.000", "0.000"),
  N_Trials           = c(876, 876, 876)
)

angle_naive_grob <- tableGrob(
  data.frame(
    "Axis"               = angle_naive_data$Axis,
    "Constant\nBaseline" = angle_naive_data$Constant_Baseline,
    "Training Range\n(°)" = angle_naive_data$Training_Range,
    "Global\nRMSE (°)"   = angle_naive_data$Global_RMSE,
    "Global\nnRMSE"      = angle_naive_data$Global_nRMSE,
    "Global\nCorr"       = angle_naive_data$Global_Corr,
    "N Trials"           = angle_naive_data$N_Trials,
    check.names = FALSE
  ),
  rows = NULL,
  theme = ttheme_default(
    core = list(
      fg_params = list(fontsize = 11),
      bg_params = list(fill = c("#f7f7f7", "#ffffff", "#f7f7f7"), col = "grey80")
    ),
    colhead = list(
      fg_params = list(fontsize = 12, fontface = "bold", col = "white"),
      bg_params = list(fill = "#2c3e50", col = "grey80")
    )
  )
)

angle_naive_title <- textGrob(
  "True Naive Baseline Results — Angles Per Axis",
  gp = gpar(fontsize = 15, fontface = "bold")
)
angle_naive_subtitle <- textGrob(
  "Baseline predicts a single constant value (overall training mean) for every timepoint — flat horizontal line.\nCorrelation is 0 by definition. Units: °",
  gp = gpar(fontsize = 10, col = "grey40")
)

png("C:/Users/aasmi/smith-dl-imu/Notes/naive_baseline_angle_table.png", width = 1100, height = 320, res = 120)
grid.arrange(
  angle_naive_title,
  angle_naive_subtitle,
  angle_naive_grob,
  heights = c(0.15, 0.12, 0.73)
)
dev.off()

message("Saved: C:/Users/aasmi/smith-dl-imu/Notes/naive_baseline_angle_table.png")


# ==============================================================================
# PyramidAttnCNN v3 — Knee Joint Moment Results (Train & Test)
# ==============================================================================

train_data <- data.frame(
  Axis         = c("X", "Y", "Z", "AVG"),
  PerTrial_Corr = c("0.966 ± 0.044", "0.946 ± 0.052", "0.970 ± 0.033", "—"),
  Global_Corr  = c("0.956", "0.935", "0.958", "0.950"),
  Global_RMSE  = c("0.711", "0.512", "0.212", "0.478"),
  Global_nRMSE = c("4.29%", "5.02%", "4.06%", "4.46%"),
  PerTrial_nRMSE = c("3.99% ± 1.59%", "4.68% ± 1.81%", "3.80% ± 1.43%", "—"),
  SD_Ratio     = c("0.636", "0.682", "0.653", "0.657"),
  N_Trials     = c("3504/3504", "3504/3504", "3504/3504", "—")
)

test_data <- data.frame(
  Axis         = c("X", "Y", "Z", "AVG"),
  PerTrial_Corr = c("0.926 ± 0.116", "0.918 ± 0.072", "0.945 ± 0.064", "—"),
  Global_Corr  = c("0.855", "0.844", "0.887", "0.862"),
  Global_RMSE  = c("1.267", "0.756", "0.339", "0.787"),
  Global_nRMSE = c("7.65%", "7.40%", "6.50%", "7.18%"),
  PerTrial_nRMSE = c("6.84% ± 3.42%", "6.70% ± 3.15%", "5.98% ± 2.55%", "—"),
  SD_Ratio     = c("0.447", "0.499", "0.453", "0.466"),
  N_Trials     = c("876/876", "876/876", "876/876", "—")
)

make_model_table <- function(df, row_fill, rmse_unit = "") {
  rmse_header <- if (nchar(rmse_unit) > 0) paste0("Global RMSE\n(", rmse_unit, ")") else "Global\nRMSE"
  tbl <- data.frame(
    "Axis"               = df$Axis,
    "Per-Trial\nCorr (μ±σ)"  = df$PerTrial_Corr,
    "Global\nCorr"       = df$Global_Corr,
    "RMSE_PLACEHOLDER"   = df$Global_RMSE,
    "Global\nnRMSE"      = df$Global_nRMSE,
    "Per-Trial\nnRMSE (μ±σ)" = df$PerTrial_nRMSE,
    "SD\nRatio"          = df$SD_Ratio,
    "N Trials"           = df$N_Trials,
    check.names = FALSE
  )
  names(tbl)[names(tbl) == "RMSE_PLACEHOLDER"] <- rmse_header
  tableGrob(
    tbl,
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

train_table <- make_model_table(train_data, train_fill, rmse_unit = "Nm/BW\u00b7Ht")
test_table  <- make_model_table(test_data,  test_fill,  rmse_unit = "Nm/BW\u00b7Ht")

main_title <- textGrob(
  "PyramidAttnCNN v3 — Knee Joint Moment Prediction Results",
  gp = gpar(fontsize = 15, fontface = "bold")
)
main_subtitle <- textGrob(
  "3-axis (X, Y, Z) knee joint moments predicted from IMU data. Global RMSE in Nm/BW\u00b7Ht; nRMSE normalized by per-axis training range.",
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


# ==============================================================================
# anglePyramidCNN RESIDUAL v3 — Knee Joint Angle Results (Train & Test)
# ==============================================================================

angle_train_data <- data.frame(
  Axis           = c("X", "Y", "Z", "AVG"),
  PerTrial_Corr  = c("0.945 ± 0.056", "0.658 ± 0.278", "0.552 ± 0.283", "—"),
  Global_Corr    = c("0.856", "0.894", "0.754", "0.835"),
  Global_RMSE    = c("5.443°", "2.208°", "4.159°", "3.937°"),
  Global_nRMSE   = c("7.16%", "7.94%", "7.49%", "7.53%"),
  PerTrial_nRMSE = c("6.39% ± 3.23%", "7.21% ± 3.34%", "6.81% ± 3.12%", "—"),
  SD_Ratio       = c("0.477", "0.836", "0.670", "0.661"),
  N_Trials       = c("3504/3504", "3504/3504", "3504/3504", "—")
)

angle_test_data <- data.frame(
  Axis           = c("X", "Y", "Z", "AVG"),
  PerTrial_Corr  = c("0.938 ± 0.060", "0.620 ± 0.294", "0.489 ± 0.295", "—"),
  Global_Corr    = c("0.778", "0.620", "0.169", "0.523"),
  Global_RMSE    = c("6.636°", "3.891°", "6.546°", "5.691°"),
  Global_nRMSE   = c("8.73%", "14.00%", "11.78%", "11.51%"),
  PerTrial_nRMSE = c("7.64% ± 4.24%", "11.80% ± 7.53%", "10.24% ± 5.84%", "—"),
  SD_Ratio       = c("0.293", "0.603", "0.456", "0.451"),
  N_Trials       = c("876/876", "876/876", "876/876", "—")
)

angle_train_fill <- c("#eaf4fb", "#ffffff", "#eaf4fb", "#d0e8f5")
angle_test_fill  <- c("#fef9e7", "#ffffff", "#fef9e7", "#fdebd0")

angle_train_table <- make_model_table(angle_train_data, angle_train_fill, rmse_unit = "\u00b0")
angle_test_table  <- make_model_table(angle_test_data,  angle_test_fill,  rmse_unit = "\u00b0")

angle_main_title <- textGrob(
  "anglePyramidCNN Residual v3 — Knee Joint Angle Prediction Results",
  gp = gpar(fontsize = 15, fontface = "bold")
)
angle_main_subtitle <- textGrob(
  "3-axis (X, Y, Z) knee joint angles predicted from IMU data. Metrics normalized by per-axis training range.",
  gp = gpar(fontsize = 10, col = "grey40")
)

angle_train_label <- textGrob("Training Set  (n = 3,504 trials)",
                              gp = gpar(fontsize = 12, fontface = "bold", col = "#1a6fa8"))
angle_test_label  <- textGrob("Test Set  (n = 876 trials)",
                              gp = gpar(fontsize = 12, fontface = "bold", col = "#b7770d"))

png("C:/Users/aasmi/smith-dl-imu/Notes/v3_angle_results_table.png",
    width = 1400, height = 560, res = 120)
grid.arrange(
  angle_main_title,
  angle_main_subtitle,
  angle_train_label,
  angle_train_table,
  angle_test_label,
  angle_test_table,
  heights = c(0.08, 0.06, 0.06, 0.32, 0.06, 0.32),
  padding = unit(0.5, "line")
)
dev.off()

message("Saved: C:/Users/aasmi/smith-dl-imu/Notes/v3_angle_results_table.png")
