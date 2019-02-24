
##########
# Analysis of the output files 
#

setwd("~/projects/FrozenLake/outputs")
require(ggplot2)
require(dplyr)

df <- read.csv('2019223_16_53_Initial_test.csv')
df$Outcome <- sapply(df$Outcome,function(x) sub("'",'',sub("b'",'',x)))
df$SMA5Steps <- SMA(df$Steps,n=5)
df$SMA10Steps <- SMA(df$Steps,n=10)

# Scatter of episode and number of steps
ggplot(df) + aes(x=`Episode`,y=`Steps`,color=`Outcome`) + geom_point()

# Scatter of episode and total reward
ggplot(df) + aes(x=`Episode`,y=`Total_Reward`,color=`Outcome`) + geom_point()

# Scatter plot episode against 5 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA5Steps,color=`Outcome`) + geom_point()

# Scatter plot episode against 10 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA10Steps,color=`Outcome`) + geom_point()

# Line plot episode against 10 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA10Steps) + geom_line()

# Line plot episode against 10 episode moving average of steps
df_goals <- df %>% filter(df$Outcome=='G')
ggplot(df_goals) + aes(x=`Episode`,y=SMA10Steps) + geom_line() + theme_classic()



