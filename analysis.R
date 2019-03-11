
##########
# Analysis of the output files 
#

setwd("~/projects/FrozenLake/outputs")
require(ggplot2)
require(dplyr)
require(TTR)
require(reshape2)
require(stringr)

# Read in data from experiment 1
#df <- read.csv('201932_11_54_Initial_experiment.csv')
df <- read.csv('Initial_experiment.csv')

# Feature engineering - add in moving averages to the data for plotting and bins of 100 episodes
# for analysis of different stages
feat_eng <- function(df){
  df$Outcome <- sapply(df$Outcome,function(x) sub("'",'',sub("b'",'',x)))
  df$SMA5Steps <- SMA(df$Steps,n=5)
  df$SMA10Steps <- SMA(df$Steps,n=10)
  df$SMA5TotalReward <- SMA(df$Total_Reward,n=5)
  df$SMA10TotalReward <- SMA(df$Total_Reward,n=10)
  df$SMA5RandomSteps <- SMA(df$Steps_random,n=5)
  df$SMA10RandomSteps <- SMA(df$Steps_random,n=10)
  episode_max <- max(df$Episode)
  episode_min <- min(df$Episode)
  df$Episode_cent <- cut(df$Episode,breaks=seq(episode_min,episode_max+1,100),include.lowest = TRUE)
  return(df)
  }

df <- feat_eng(df)

# Function to apply feature engineering to each run of episodes (needs to be separate)
feat_by_param <- function(df,param1,param2){
  dfs <- split(df,c(param1,param2))
  dfs %>% sapply(feat_eng)
  return(rbind(dfs))
  }

add_param_label<-function(df,param,value){
  df$param_col <- value
  names(df) <- gsub('param_col',param,names(df))
  return(df)
}

########################
# Exploratory charts for experiment 1

# Scatter of episode and number of steps
ggplot(df) + aes(x=`Episode`,y=`Steps`,color=`Outcome`) + geom_point() + theme_light()

# Scatter of episode and total reward
ggplot(df) + aes(x=`Episode`,y=`Total_Reward`,color=`Outcome`) + geom_point()

# Line plot of episode and total reward
ggplot(df) + aes(x=`Episode`,y=`Total_Reward`) + geom_line()

# Scatter plot episode against 10 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA10Steps,color=`Outcome`) + geom_point()

# Line plot episode against 10 episode moving average of steps
ggplot(df) + aes(x=`Episode`,y=SMA10Steps) + geom_line() + theme_light()

# Line plot episode against 10 episode moving average of steps
df_goals <- df %>% filter(df$Outcome=='G')
ggplot(df_goals) + aes(x=`Episode`,y=SMA10Steps) + geom_line() + theme_classic()

# Line plot episode against 5 episode moving average of total reward
ggplot(df) + aes(x=`Episode`,y=SMA5TotalReward) + geom_line() + theme_light()

# Epsilon decay
ggplot(df) + aes(x=`Episode`,y=Epsilon_start) + geom_line() + theme_light()

# Quantitative analysis of experiment 1
df %>% select(Episode_cent,Total_Reward) %>% group_by(Episode_cent) %>% 
  summarise(av_total_reward = mean(Total_Reward),var_total_reward = var(Total_Reward))

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

# Get a subset of data with only average step metrics and unpivot
steps <- df %>% select(Episode,SMA10Steps,SMA10RandomSteps) %>% melt(id=('Episode'))

# Line plot episode against 10 episode moving average of steps and random steps
ggplot(steps) + aes(x=`Episode`,y=value,color=variable) + geom_line() + ylab("Moving Average Number of Steps in episode (n=10)") +
  scale_color_hue(labels = c("Total Steps", "Random Steps"),name="Metric") +
  theme_light()
ggsave(filename="../plots/line_episode_steps_random_sma.jpg", plot=last_plot(),width=7,height=4,units="in")


########################
# Analysis of varying decay factors

# Load CSV files from decay factor experiment
file_list <- list.files(pattern="experiment_dfs_.*.csv")
params <- lapply(file_list,function(x) str_match(x,"experiment_dfs_(\\d\\.\\d{6})_(\\d\\.\\d{6})\\.csv")[,2:3])
dfs <- lapply(file_list, read.csv)
dfs <- lapply(dfs,feat_eng)
for(i in 1:length(dfs)){
  dfs[[i]]$df1 <- params[[i]][1]
  dfs[[i]]$df2 <- params[[i]][2]
}

decfac <- bind_rows(dfs)

av_reward_400_500 <- decfac %>% filter(Episode_cent=='(400,500]') %>% select(df1,df2,Total_Reward) %>%
  group_by(df1,df2) %>% summarise(av_total_reward = mean(Total_Reward))

av_reward_300_400 <- decfac %>% filter(Episode_cent=='(300,400]') %>% select(df1,df2,Total_Reward) %>%
  group_by(df1,df2) %>% summarise(av_total_reward = mean(Total_Reward))

av_reward_200_300 <- decfac %>% filter(Episode_cent=='(200,300]') %>% select(df1,df2,Total_Reward) %>%
  group_by(df1,df2) %>% summarise(av_total_reward = mean(Total_Reward))

ggplot(av_reward_400_500, aes(df1, df2)) + geom_tile(aes(fill = av_total_reward),colour = "white") + 
  scale_fill_gradient(low = "white",high = "steelblue",name='Mean Total Reward')
ggsave(filename="../plots/heatmap_400_500_df1df2.png", plot=last_plot(),width=2.5,height=2.5,units="in")

ggplot(av_reward_300_400, aes(df1, df2)) + geom_tile(aes(fill = av_total_reward),colour = "white") + 
  scale_fill_gradient(low = "white",high = "steelblue",name='Mean Total Reward')
ggsave(filename="../plots/heatmap_300_400_df1df2.png", plot=last_plot(),width=2.5,height=2.5,units="in")

ggplot(av_reward_200_300, aes(df1, df2)) + geom_tile(aes(fill = av_total_reward),colour = "white") + 
  scale_fill_gradient(low = "white",high = "steelblue",name='Mean Total Reward')

# Looks from the heatmap like it's better to have a lower value of df1. Plot the total reward for these parameters.

decfac_plot <- decfac %>% select(Episode,SMA10TotalReward,df1,df2) %>% filter(df2 %in% c('0.900000','0.990000'))
decfac_plot$df_comb <- sprintf('df1=%s, df2=%s',decfac_plot$df1,decfac_plot$df2)

ggplot(decfac_plot) + aes(x=`Episode`,y=SMA10TotalReward,colour=df_comb,shape=df_comb) + 
  geom_line() + geom_point(size=2) +
  scale_shape_manual(values=1:12,name=c('Decay Factors')) +
  xlim(c(0,250)) +
  ylab("Moving average (n=10) total reward") + 
  scale_color_discrete(name=c('Decay Factors')) + 
  theme_light()
ggsave(filename="../plots/line_episode_reward_sma_df1df2.jpg", plot=last_plot(),width=7,height=4,units="in")


decfac_eps <- decfac %>% select(Episode,Epsilon_start,Steps_random,df1,df2) %>% filter(df2 %in% c('0.900000','0.990000'))
decfac_eps$df_comb <- sprintf('df1=%s, df2=%s',decfac_eps$df1,decfac_eps$df2)

ggplot(decfac_eps) + aes(x=`Episode`,y=Epsilon_start,colour=df_comb,shape=df_comb) + 
  geom_line() + geom_point(size=2) +
  scale_shape_manual(values=1:12,name=c('Decay Factors')) +
  xlim(c(0,250)) +
  ylab("Moving average (n=10) total reward") + 
  scale_color_discrete(name=c('Decay Factors')) + 
  theme_light()

ggplot(decfac_eps %>% filter(df1=='0.900000'& df2=='0.900000' |  df1=='0.900000'& df2=='0.990000' | df1=='0.900000'& df2=='0.999000')) + 
  aes(x=`Episode`,y=Epsilon_start,colour=df_comb,alpha=Steps_random) + geom_line(size=3) +
  xlim(c(0,10)) +
  scale_alpha_continuous(name='Random Steps') +
  scale_color_discrete(name='Decay factors') +
  #scale_x_continuous(trans='log10') +
  ylab("Epsilon at episode start") + 
  theme_light()
ggsave(filename="../plots/line_epsilon_decay_select_dfs.jpg", plot=last_plot(),width=7,height=4,units="in")

########################
# Analysis of varying gamma experiment

#gm_filename <- '201932_16_46_experiment_gamma_'
#gm_filename <- '201932_19_19_experiment_gamma_'
gm_filename <- '201937_12_2_experiment_gamma_'

gm01 <- read.csv(sprintf('%s%2.1f.csv',gm_filename,0.1)) %>% feat_eng() %>% add_param_label('gamma',0.1)
gm03 <- read.csv(sprintf('%s%2.1f.csv',gm_filename,0.3)) %>% feat_eng() %>% add_param_label('gamma',0.3)
gm05 <- read.csv(sprintf('%s%2.1f.csv',gm_filename,0.5)) %>% feat_eng() %>% add_param_label('gamma',0.5)
gm07 <- read.csv(sprintf('%s%2.1f.csv',gm_filename,0.7)) %>% feat_eng() %>% add_param_label('gamma',0.7)
gm09 <- read.csv(sprintf('%s%2.1f.csv',gm_filename,0.9)) %>% feat_eng() %>% add_param_label('gamma',0.9)

gm <- rbind(gm01,gm03,gm05,gm07,gm09)

gm$gamma <- as.factor(gm$gamma)

# Line plot episode against 10 episode moving average of total reward for each gamma
ggplot(gm) + aes(x=`Episode`,y=SMA10TotalReward,colour=gamma) + geom_line() + 
  ylab("Moving average (n=10) total reward") + scale_color_discrete() + theme_light()
ggsave(filename="../plots/line_episode_reward_sma_gamma.jpg", plot=last_plot(),width=7,height=4,units="in")

# Quantitative analysis
gm01 %>% summary()

gm %>% select(Episode_cent,Total_Reward,gamma) %>% group_by(Episode_cent,gamma) %>% 
  summarise(av_total_reward = mean(Total_Reward)) %>% dcast(Episode_cent ~ gamma) %>% 

########################
# Analysis of varying alpha experiment

#al_filename = '201932_20_2_experiment_alpha_'
al_filename = '201932_20_2_experiment_alpha_'

al01 <- read.csv(sprintf('%s%2.1f.csv',al_filename,0.1)) %>% feat_eng() %>% add_param_label('alpha',0.1)
al03 <- read.csv(sprintf('%s%2.1f.csv',al_filename,0.3)) %>% feat_eng() %>% add_param_label('alpha',0.3)
al05 <- read.csv(sprintf('%s%2.1f.csv',al_filename,0.5)) %>% feat_eng() %>% add_param_label('alpha',0.5)

al <- rbind(al01,al03,al05)
al$alpha <- as.factor(al$alpha)

# Line plot episode against 10 episode moving average of total reward for each alpha
ggplot(al) + aes(x=`Episode`,y=SMA10TotalReward,colour=alpha) + geom_line() + 
  ylab("Moving average (n=10) total reward") + scale_color_discrete() + theme_light()
ggsave(filename="../plots/line_episode_reward_sma_alpha.jpg", plot=last_plot(),width=7,height=4,units="in")








