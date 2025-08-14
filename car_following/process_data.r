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
setwd("C:/Users/samil/Documents/MODL/car_following")
data = read.csv("outputs/test_constrained.csv")

data_x = data[,grep("x",names(data))]
data_x$time = data$X
data_y = data[,grep("y",names(data))]
data_y$time = data$X
data_v = data[,grep("v",names(data))]
data_v$time = data$X
data_theta = data[,grep("heading",names(data))]
data_theta$time = data$X
data_steering = data[,grep("steering_angle",names(data))]
data_steering$time = data$X
data_acceleration = data[,grep("acceleration",names(data))]
data_acceleration$time = data$X
data_steering_angle_rate = data[,grep("steering_angle_rate",names(data))]
data_steering_angle_rate$time = data$X

data_distance = data[,grep("x|y",names(data))]
data_distance$time = data$X



cols_to_plot <- names(data_x)[10:20]  # or your desired columns

plot_x = ggplot(data_x,aes(x=time/10))
for (cols in cols_to_plot){
  plot_x = plot_x + geom_point(aes(y =.data[[cols]]),color='blue')
}

cols_to_plot <- names(data_y)[10:20]
plot_y = ggplot(data_y,aes(x=time/10))
for (cols in cols_to_plot){
  plot_y = plot_y + geom_point(aes(y =.data[[cols]]),color='darkgreen')
}

cols_to_plot <- names(data_v)[10:20]
plot_v = ggplot(data_v,aes(x=time/10))
for (cols in cols_to_plot){
  plot_v = plot_v + geom_point(aes(y =.data[[cols]]),color='purple')
}

cols_to_plot <- names(data_theta)[10:20]
plot_theta = ggplot(data_theta,aes(x=time/10))
for (cols in cols_to_plot){
  plot_theta = plot_theta + geom_point(aes(y =.data[[cols]]),color='purple')
}

cols_to_plot <- names(data_steering)[10:20]
plot_steering = ggplot(data_steering,aes(x=time/10))
for (cols in cols_to_plot){
  plot_steering = plot_steering + geom_line(aes(y =.data[[cols]]),color='purple')
}

cols_to_plot <- names(data_acceleration)[10:20]
plot_acceleration = ggplot(data_acceleration,aes(x=time/10))
for (cols in cols_to_plot){
  plot_acceleration = plot_acceleration + geom_line(aes(y =.data[[cols]]),color='purple')
}

cols_to_plot <- names(data_steering_angle_rate)[10:15]
plot_steering_rate = ggplot(data_steering_angle_rate,aes(x=time/10))
for (cols in cols_to_plot){
  plot_steering_rate = plot_steering_rate + geom_line(aes(y =.data[[cols]]),color='purple')
}

cols_to_plot <- names(data_distance)[20:40]  # or your desired columns

plot_distance = ggplot(data_distance,aes(x=time/10))

for (i in seq(1, 20, by = 2)) {
  pair <- c(i, i+1)
  plot_distance = plot_distance + geom_point(aes(y =(.data[[cols_to_plot[pair[1]]]]**2+
                                                   .data[[cols_to_plot[pair[2]]]]**2)**(1/2)),color='blue')
}



prev= plot_distance/ plot_v / plot_theta
/ plot_steering


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

