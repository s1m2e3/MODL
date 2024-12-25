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

setwd("C:/Users/samil/Documents/MODL/outputs")
results = read.csv('losses.csv')
results_non = results[,c("X",'first_non',"second_non")]
results_non$kind = "Regular Gradient Descent"
colnames(results_non)=c("X","First_Objective","Second_Objective","Kind")
results_first = results[,c("X",'first_extra',"second_extra")]
results_first$kind = "Multi-Objective Gradient Descent"
colnames(results_first)=c("X","First_Objective","Second_Objective","Kind")

results = rbind(results_first,results_non)

results$X_cuts = cut(results$X,breaks = seq(min(results$X), max(results$X), by = 100)
)
ggplot(data=results%>%filter(X%%10==0))+geom_point(aes(x=First_Objective,y=Second_Objective,color=X))+
  scale_x_log10()+scale_y_log10()+
  scale_color_gradient(low="green",high="red")+
  labs(title="Loss Function Evolution",
       color = "Iteration Number",
       x="First Objective Function - LogScale",y="Second Objective Function - LogScale")+
  theme_minimal()+facet_grid(~Kind) +
  theme(
    plot.title = element_text(size = 12, face = "bold", family = "serif"), # Title font
    axis.title.x = element_text(size = 10, face = "italic", family = "serif"), # X-axis label font
    axis.title.y = element_text(size = 10, face = "italic", family = "serif"),  # Y-axis label font
    axis.text.y = element_text(size = 8, face = "italic", family = "serif"),
    axis.text.x = element_text(size = 8, face = "italic", family = "serif"),
    # Customize grid lines
    panel.grid.major = element_blank(),      # Remove all major grid lines
    panel.grid.minor = element_blank(),      # Remove all minor grid lines
    panel.grid.major.x = element_line(color = "black", linetype="dashed"),   # Show only main vertical line
    panel.grid.major.y = element_line(color = "black", linetype="dashed"),    # Show only main horizontal line
    panel.background = element_rect(fill = "snow", color = NA),
  )

  
#   geom_point(data=results%>%filter(X>100),aes(x=second_non,y=first_non,color=X_cuts),shape="x",size=3)+ylim(0,0.10)+xlim(0,0.1)
#   


