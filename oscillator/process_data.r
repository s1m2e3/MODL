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
setwd("C:/Users/samil/Documents/MODL/oscillator")
data = read.csv("outputs/test_clean.csv")

data_x = data[,grep("x",names(data))]
data_x$time = data$X
data_v = data[,grep("v",names(data))]
data_v$time = data$X
data_u = data[,grep("u",names(data))]
data_u$time = data$X
cols_to_plot <- names(data_x)[1:30]  # or your desired columns

plot_x = ggplot(data_x,aes(x=time/10))
for (cols in cols_to_plot){
  plot_x = plot_x + geom_point(aes(y =.data[[cols]]),color='blue')
}

cols_to_plot <- names(data_v)[1:30]
plot_v = ggplot(data_v,aes(x=time/10))
for (cols in cols_to_plot){
  plot_v = plot_v + geom_point(aes(y =.data[[cols]]),color='darkgreen')
}

cols_to_plot <- names(data_u)[1:30]
plot_u = ggplot(data_u,aes(x=time/10))
for (cols in cols_to_plot){
  plot_u = plot_u + geom_point(aes(y =.data[[cols]]),color='purple')
}

prev= plot_x / plot_v / plot_u


data = read.csv("outputs/test_constrained.csv")

data_x = data[,grep("x",names(data))]
data_x$time = data$X
data_v = data[,grep("v",names(data))]
data_v$time = data$X
data_u = data[,grep("u",names(data))]
data_u$time = data$X
cols_to_plot <- names(data_x)[1:30]  # or your desired columns

plot_x = ggplot(data_x,aes(x=time/10))
for (cols in cols_to_plot){
  plot_x = plot_x + geom_point(aes(y =.data[[cols]]),color='blue',alpha=0.5)
}

cols_to_plot <- names(data_v)[1:30]
plot_v = ggplot(data_v,aes(x=time/10))
for (cols in cols_to_plot){
  plot_v = plot_v + geom_point(aes(y =.data[[cols]]),color='darkgreen',alpha=0.5)
}

cols_to_plot <- names(data_u)[1:30]
plot_u = ggplot(data_u,aes(x=time/10))
for (cols in cols_to_plot){
  plot_u = plot_u + geom_point(aes(y =.data[[cols]]),color='purple',alpha=0.5)
}

post=plot_x / plot_v / plot_u

