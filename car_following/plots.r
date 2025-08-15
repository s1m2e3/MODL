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
data_clean <- read.csv("rnn_data_clean.csv")
data_guided <- read.csv("rnn_data_constrained.csv")

data_clean$trajectory_step = as.numeric(data_clean$trajectory_step)
data_guided$trajectory_iteration = as.numeric(data_guided$trajectory_iteration)

data_clean$trajectory_step = as.numeric(data_clean$trajectory_step)
data_guided$trajectory_iteration = as.numeric(data_guided$trajectory_iteration)

data_clean = data_clean%>%filter(trajectory_iteration %% 10 == 0 | trajectory_iteration == 99)
data_guided = data_guided%>%filter(trajectory_iteration %% 10 == 0 | trajectory_iteration == 99)

data_clean$source = "unguided"  
data_guided$source = "guided"

df = rbind(data_clean,data_guided)

df_states = df %>% filter(concept == 'x' | concept == 'u' |concept == 'u_init' | concept == 'u_final')
df_states$element = paste(df_states$concept, df_states$vector_index, sep = "_")

wide_df_states = pivot_wider(
  data =df_states,
  names_from =element,
  values_from = V1,
  id_cols = c(trajectory_iteration,trajectory_step,source)
)

wide_df_states$trajectory_step = as.numeric(wide_df_states$trajectory_step)
wide_df_states = wide_df_states %>% filter(trajectory_iteration == 0 |trajectory_iteration == 10|trajectory_iteration == 20|trajectory_iteration == 30|trajectory_iteration == 40|
                                             trajectory_iteration == 50| trajectory_iteration == 90|trajectory_iteration == 99) 
wide_df_states <- wide_df_states %>%
  arrange(trajectory_iteration, trajectory_step)

plot_positions = ggplot(wide_df_states)+geom_line(aes(x=x_1,y=x_2,color=as.factor(trajectory_iteration)),linewidth=2)+facet_wrap(~source)+
  scale_color_brewer(palette="Set3")
plot_distance = ggplot(wide_df_states)+geom_line(aes(x=as.numeric(trajectory_step),y=sqrt((x_1-15)**2+(x_2-15)**2),color=as.factor(trajectory_iteration)),linewidth=2)+facet_wrap(~source)+
  scale_color_brewer(palette="Set3")
plot_angle_distance = ggplot(wide_df_states)+geom_line(aes(x=as.numeric(trajectory_step),y=x_3-0.78,color=as.factor(trajectory_iteration)),linewidth=2)+facet_wrap(~source)+
  scale_color_brewer(palette="Set3")
plot_speed = ggplot(wide_df_states)+geom_line(aes(x=as.numeric(trajectory_step),y=u_1,color=as.factor(trajectory_iteration)),linewidth=2)+facet_wrap(~source)+
  scale_color_brewer(palette="Set3")
plot_angle = ggplot(wide_df_states)+geom_line(aes(x=as.numeric(trajectory_step),y=u_2,color=as.factor(trajectory_iteration)),linewidth=2)+facet_wrap(~source)+
  scale_color_brewer(palette="Set3")
plot_speed_init = ggplot(wide_df_states%>%filter(source=="guided"))+geom_line(aes(x=as.numeric(trajectory_step),y=u_init_1,color=as.factor(trajectory_iteration)),linewidth=2)+
  scale_color_brewer(palette="Set3")
plot_angle_init = ggplot(wide_df_states%>%filter(source=="guided"))+geom_line(aes(x=as.numeric(trajectory_step),y=u_init_2,color=as.factor(trajectory_iteration)),linewidth=2)+
  scale_color_brewer(palette="Set3")
plot_speed_change = ggplot(wide_df_states)+geom_line(aes(x=as.numeric(trajectory_step),y=u_init_1-u_final_1,color=as.factor(trajectory_iteration)),linewidth=2)+
  scale_color_brewer(palette="Set3")+facet_wrap(~source)
plot_angle_change = ggplot(wide_df_states)+geom_line(aes(x=as.numeric(trajectory_step),y=u_init_2-u_final_2,color=as.factor(trajectory_iteration)),linewidth=2)+
  scale_color_brewer(palette="Set3")+facet_wrap(~source)

for (cols in cols_to_plot){
  plot_x = plot_x + geom_point(aes(y =.data[[cols]]),color=my_colors[1])
}
plot_x = plot_x + labs(title="Controlled Harmonic Oscillator ", y= "Position (m)",color="Position")+
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", family = "serif"), # Title font
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 14, face = "italic", family = "serif"),  # Y-axis label font
    axis.text.y = element_text(size = 12, face = "italic", family = "serif"),
    axis.text.x = element_text(size = 12, face = "italic", family = "serif"),
    strip.text=element_text(size=14,face="italic",family='serif'),
    # Customize grid lines
    panel.grid.major = element_blank(),      # Remove all major grid lines
    panel.grid.minor = element_blank(),      # Remove all minor grid lines
    panel.grid.major.x = element_line(color = "black", linetype="dashed"),   # Show only main vertical line
    panel.grid.major.y = element_line(color = "black", linetype="dashed"),    # Show only main horizontal line
    panel.background = element_rect(fill = "snow", color = NA),
  )+xlim(0,25)


cols_to_plot <- names(data_v)[1:10]
plot_v = ggplot(data_v,aes(x=time/10))
for (cols in cols_to_plot){
  plot_v = plot_v + geom_point(aes(y =.data[[cols]]),color=my_colors[2])
}

plot_v = plot_v + labs(y= "Velocity (m/2)",color="Velocity")+
  theme_minimal() +
  theme(
    plot.title = element_blank(), # Title font
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 14, face = "italic", family = "serif"),  # Y-axis label font
    axis.text.y = element_text(size = 12, face = "italic", family = "serif"),
    axis.text.x = element_text(size = 12, face = "italic", family = "serif"),
    strip.text=element_text(size=14,face="italic",family='serif'),
    # Customize grid lines
    panel.grid.major = element_blank(),      # Remove all major grid lines
    panel.grid.minor = element_blank(),      # Remove all minor grid lines
    panel.grid.major.x = element_line(color = "black", linetype="dashed"),   # Show only main vertical line
    panel.grid.major.y = element_line(color = "black", linetype="dashed"),    # Show only main horizontal line
    panel.background = element_rect(fill = "snow", color = NA),
  )+xlim(0,25)


cols_to_plot <- names(data_u)[1:10]
plot_u = ggplot(data_u,aes(x=time/10))
for (cols in cols_to_plot){
  plot_u = plot_u + geom_point(aes(y =.data[[cols]]),color=my_colors[3])
}

plot_u = plot_u + labs(y= "Force (N)",color="Force",x='Time (s)')+
  theme_minimal() +
  theme(
    plot.title = element_blank(), # Title font
    axis.title.x = element_text(size = 14, face = "italic", family = "serif"),
    axis.title.y = element_text(size = 14, face = "italic", family = "serif"),  # Y-axis label font
    axis.text.y = element_text(size = 12, face = "italic", family = "serif"),
    axis.text.x = element_text(size = 12, face = "italic", family = "serif"),
    strip.text=element_text(size=14,face="italic",family='serif'),
    # Customize grid lines
    panel.grid.major = element_blank(),      # Remove all major grid lines
    panel.grid.minor = element_blank(),      # Remove all minor grid lines
    panel.grid.major.x = element_line(color = "black", linetype="dashed"),   # Show only main vertical line
    panel.grid.major.y = element_line(color = "black", linetype="dashed"),    # Show only main horizontal line
    panel.background = element_rect(fill = "snow", color = NA),
  )+xlim(0,25)

prev= plot_x / plot_v / plot_u
ggsave("oscillator_clean.png",prev)


data = read.csv("outputs/test_constrained.csv")

data_x = data[,grep("x",names(data))]
data_x$time = data$X
data_v = data[,grep("v",names(data))]
data_v$time = data$X
data_u = data[,grep("u",names(data))]
data_u$time = data$X
cols_to_plot <- names(data_x)[1:10]  # or your desired columns

plot_x = ggplot(data_x,aes(x=time/10))
for (cols in cols_to_plot){
  plot_x = plot_x + geom_point(aes(y =.data[[cols]]),color=my_colors[1])
}
plot_x = plot_x + labs(title="Constrained Controlled Harmonic Oscillator ", y= "Position (m)",color="Position")+
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold", family = "serif"), # Title font
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 14, face = "italic", family = "serif"),  # Y-axis label font
    axis.text.y = element_text(size = 12, face = "italic", family = "serif"),
    axis.text.x = element_text(size = 12, face = "italic", family = "serif"),
    strip.text=element_text(size=14,face="italic",family='serif'),
    # Customize grid lines
    panel.grid.major = element_blank(),      # Remove all major grid lines
    panel.grid.minor = element_blank(),      # Remove all minor grid lines
    panel.grid.major.x = element_line(color = "black", linetype="dashed"),   # Show only main vertical line
    panel.grid.major.y = element_line(color = "black", linetype="dashed"),    # Show only main horizontal line
    panel.background = element_rect(fill = "snow", color = NA),
  )+xlim(0,60)

cols_to_plot <- names(data_v)[1:10]
plot_v = ggplot(data_v,aes(x=time/10))
for (cols in cols_to_plot){
  plot_v = plot_v + geom_point(aes(y =.data[[cols]]),color=my_colors[2])
}
plot_v = plot_v + labs(y= "Velocity (m/2)",color="Velocity")+
  theme_minimal() +
  theme(
    plot.title = element_blank(), # Title font
    axis.title.x = element_blank(),
    axis.title.y = element_text(size = 14, face = "italic", family = "serif"),  # Y-axis label font
    axis.text.y = element_text(size = 12, face = "italic", family = "serif"),
    axis.text.x = element_text(size = 12, face = "italic", family = "serif"),
    strip.text=element_text(size=14,face="italic",family='serif'),
    # Customize grid lines
    panel.grid.major = element_blank(),      # Remove all major grid lines
    panel.grid.minor = element_blank(),      # Remove all minor grid lines
    panel.grid.major.x = element_line(color = "black", linetype="dashed"),   # Show only main vertical line
    panel.grid.major.y = element_line(color = "black", linetype="dashed"),    # Show only main horizontal line
    panel.background = element_rect(fill = "snow", color = NA),
  )+xlim(0,60)

cols_to_plot <- names(data_u)[1:10]
plot_u = ggplot(data_u,aes(x=time/10))
for (cols in cols_to_plot){
  plot_u = plot_u + geom_point(aes(y =.data[[cols]]),color=my_colors[3])
}

plot_u = plot_u + labs(y= "Force (N)",color="Force",x='Time (s)')+
  theme_minimal() +
  theme(
    plot.title = element_blank(), # Title font
    axis.title.x = element_text(size = 14, face = "italic", family = "serif"),
    axis.title.y = element_text(size = 14, face = "italic", family = "serif"),  # Y-axis label font
    axis.text.y = element_text(size = 12, face = "italic", family = "serif"),
    axis.text.x = element_text(size = 12, face = "italic", family = "serif"),
    strip.text=element_text(size=14,face="italic",family='serif'),
    # Customize grid lines
    panel.grid.major = element_blank(),      # Remove all major grid lines
    panel.grid.minor = element_blank(),      # Remove all minor grid lines
    panel.grid.major.x = element_line(color = "black", linetype="dashed"),   # Show only main vertical line
    panel.grid.major.y = element_line(color = "black", linetype="dashed"),    # Show only main horizontal line
    panel.background = element_rect(fill = "snow", color = NA),
  )+xlim(0,60)


post=plot_x / plot_v / plot_u
ggsave("oscillator_constrained.png",post)