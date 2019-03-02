
##########
# Analysis of the output files 
#

setwd("~/projects/FrozenLake/outputs")
require(ggplot2)
require(dplyr)
require(TTR)
require(reshape2)

df <- read.csv('201932_11_54_Initial_experiment.csv')
df$Outcome <- sapply(df$Outcome,function(x) sub("'",'',sub("b'",'',x)))
df$SMA5Steps <- SMA(df$Steps,n=5)
df$SMA10Steps <- SMA(df$Steps,n=10)
df$SMA5TotalReward <- SMA(df$Total_Reward,n=5)
df$SMA10TotalReward <- SMA(df$Total_Reward,n=10)
df$SMA5RandomSteps <- SMA(df$Steps_random,n=5)
df$SMA10RandomSteps <- SMA(df$Steps_random,n=10)

########################
# Exploratory charts

# Scatter of episode and number of steps
ggplot(df) + aes(x=`Episode`,y=`Steps`,color=`Outcome`) + geom_point() + theme_light()

# Scatter of episode and total reward
ggplot(df) + aes(x=`Episode`,y=`Total_Reward`,color=`Outcome`) + geom_point()

# Scatter plot episode against 10 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA10Steps,color=`Outcome`) + geom_point()

# Line plot episode against 10 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA10Steps) + geom_line() + theme_light()

# Line plot episode against 10 episode moving average of steps
df_goals <- df %>% filter(df$Outcome=='G')
ggplot(df_goals) + aes(x=`Episode`,y=SMA10Steps) + geom_line() + theme_classic()

# Line plot episode against 5 episode moving average of total reward
ggplot(df) + aes(x=`Episode`,y=SMA5TotalReward) + geom_line() + theme_light()


# Get a subset of data with only average step metrics and unpivot
steps <-? df %>% select(Episode,SMA10Steps,SMA10RandomSteps) %>% melt(id=('Episode'))

########################
# Used in analysis

# Scatter plot episode against 5 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA5Steps,color=`Outcome`) + geom_point() +
  ylab("Steps") + theme_light()
ggsave(filename="../plots/scatter_episode_steps_sma.jpg", plot=last_plot())

# Line plot episode against 10 episode moving average of total reward
ggplot(df) + aes(x=`Episode`,y=SMA10TotalReward) + geom_line() + 
  ylab("Moving average (n=10) total reward") + theme_light()
ggsave(filename="../plots/line_episode_reward_sma.jpg", plot=last_plot())

# Line plot episode against 10 episode moving average of steps and random steps
ggplot(steps) + aes(x=`Episode`,y=value,color=variable) + geom_line() + ylab("Moving Average Number of Steps in episode (n=10)") +
  scale_color_hue(labels = c("Total Steps", "Random Steps"),name="Metric") +
  theme_light()
ggsave(filename="../plots/line_episode_steps_random_sma.jpg", plot=last_plot())





