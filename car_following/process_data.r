library(dplyr)
library(MASS)
library(ks)
library(slider)
library(ggplot2)
library(RColorBrewer)
library(ggnewscale)
library(cowplot)
library(HDInterval)
library(zoo)
library(tidyr)
library(tibble)
library(tseries)
library(forecast)
library(zoo)
library(patchwork)
library(jsonlite)
setwd("C:/Users/samil/Documents/MODL/car_following/outputs")

data_clean = fromJSON('weights_rnn.json')
df_clean = as.data.frame(t(as.data.frame(data_clean[[1]])))
df_clean$trajectory_step=0
df_clean$trajectory_iteration=0
df_clean$concept = "unknown"
df_clean$vector_index = 0

# Loop over rows
for (i in 1:nrow(df_clean)) {
  row_name <- rownames(df_clean)[i]           # get row name
  parts <- strsplit(row_name, "\\.")[[1]]  # split at "."
  df_clean$trajectory_iteration[i] = gsub("^X(?=\\d)", "", parts[1], perl = TRUE)
  df_clean$trajectory_step[i]=parts[2]
  df_clean$concept[i] = parts[3]
  if (grepl("x|u", parts[3])) {
    df_clean$vector_index[i] = parts[4]
  }
}

data_guided = fromJSON('weights_rnn_guided.json')
df_guided = as.data.frame(t(as.data.frame(data_guided[[1]])))
df_guided$trajectory_step=0
df_guided$trajectory_iteration=0
df_guided$concept = "unknown"
df_guided$vector_index = 0

# Loop over rows
for (i in 1:nrow(df_guided)) {
  row_name <- rownames(df_guided)[i]           # get row name
  parts <- strsplit(row_name, "\\.")[[1]]  # split at "."
  df_guided$trajectory_iteration[i] = gsub("^X(?=\\d)", "", parts[1], perl = TRUE)
  df_guided$trajectory_step[i]=parts[2]
  df_guided$concept[i] = parts[3]
  if (grepl("x|u", parts[3])) {
    df_guided$vector_index[i] = parts[4]
  }
}

write.csv(df_clean,"rnn_data_clean.csv")
write.csv(df_guided,"rnn_data_constrained.csv")
