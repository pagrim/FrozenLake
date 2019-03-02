
##########
# Analysis of the output files 
#

setwd("~/projects/FrozenLake/outputs")
require(ggplot2)
require(dplyr)
require(TTR)
require(reshape2)

df <- read.csv('201932_11_54_Initial_experiment.csv')

feat_eng <- function(df){
  df$Outcome <- sapply(df$Outcome,function(x) sub("'",'',sub("b'",'',x)))
  df$SMA5Steps <- SMA(df$Steps,n=5)
  df$SMA10Steps <- SMA(df$Steps,n=10)
  df$SMA5TotalReward <- SMA(df$Total_Reward,n=5)
  df$SMA10TotalReward <- SMA(df$Total_Reward,n=10)
  df$SMA5RandomSteps <- SMA(df$Steps_random,n=5)
  df$SMA10RandomSteps <- SMA(df$Steps_random,n=10)  
  return(df)
  }

df <- feat_eng(df)

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
steps <- df %>% select(Episode,SMA10Steps,SMA10RandomSteps) %>% melt(id=('Episode'))

########################
# Analysis of initial experiment

# Scatter plot episode against 5 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA5Steps,color=`Outcome`) + geom_point(size=2) +
  ylab("Steps") + theme_light()
ggsave(filename="../plots/scatter_episode_steps_sma.jpg", plot=last_plot(),width=6,height=4,units="in")

# Line plot episode against 10 episode moving average of total reward
ggplot(df) + aes(x=`Episode`,y=SMA10TotalReward) + geom_line() + 
  ylab("Moving average (n=10) total reward") + theme_light()
ggsave(filename="../plots/line_episode_reward_sma.jpg", plot=last_plot(),width=7,height=4,units="in")

# Line plot episode against 10 episode moving average of steps and random steps
ggplot(steps) + aes(x=`Episode`,y=value,color=variable) + geom_line() + ylab("Moving Average Number of Steps in episode (n=10)") +
  scale_color_hue(labels = c("Total Steps", "Random Steps"),name="Metric") +
  theme_light()
ggsave(filename="../plots/line_episode_steps_random_sma.jpg", plot=last_plot(),width=7,height=4,units="in")

########################
# Analysis of varying gamma experiment

add_param_label<-function(df,param,value){
  df$param_col <- value
  names(df) <- gsub('param_col',param,names(df))
  return(df)
}

gm01 <- read.csv('201932_16_46_experiment_gamma_0.1.csv') %>% feat_eng() %>% add_param_label('gamma',0.1)
gm03 <- read.csv('201932_16_46_experiment_gamma_0.3.csv') %>% feat_eng() %>% add_param_label('gamma',0.3)
gm05 <- read.csv('201932_16_46_experiment_gamma_0.5.csv') %>% feat_eng() %>% add_param_label('gamma',0.5)
gm07 <- read.csv('201932_16_46_experiment_gamma_0.7.csv') %>% feat_eng() %>% add_param_label('gamma',0.7)
gm09 <- read.csv('201932_16_46_experiment_gamma_0.9.csv') %>% feat_eng() %>% add_param_label('gamma',0.9)

gm <- rbind(gm01,gm03,gm05,gm07,gm09)

# Line plot episode against 10 episode moving average of total reward of each gamma value
#gm_rew <- cbind(gm01$SMA10TotalReward,gm03$SMA10TotalReward,gm03$SMA10TotalReward,gm)

#gm_rew <- gm %>% select(Episode,SMA10TotalReward,gamma) %>% melt(id='Episode')

gm <- gm %>% filter(gamma==c(0.3))

ggplot(gm) + aes(x=`Episode`,y=SMA10TotalReward,colour=gamma) + geom_line() + 
  ylab("Moving average (n=10) total reward") + scale_y_discrete() + theme_light()
ggsave(filename="../plots/line_episode_reward_sma.jpg", plot=last_plot(),width=7,height=4,units="in")

