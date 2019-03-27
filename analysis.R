
##########
# This script was used to do analysis of experiments run on the Q-learning and SARSA algorithms 

setwd("~/projects/FrozenLake/outputs")
require(ggplot2)
require(dplyr)
require(TTR)
require(reshape2)
require(stringr)

# This function creates additional features for analysis; moving averages and grouping the episodes for analysis
feat_eng <- function(df){
  df$Outcome <- sapply(df$Outcome,function(x) sub("'",'',sub("b'",'',x)))
  df$SMA5Steps <- SMA(df$Steps,n=5)
  df$SMA10Steps <- SMA(df$Steps,n=10)
  df$SMA5TotalReward <- SMA(df$Total_Reward,n=5)
  df$SMA10TotalReward <- SMA(df$Total_Reward,n=10)
  if('Steps_random' %in% colnames(df)){
    df$SMA5RandomSteps <- SMA(df$Steps_random,n=5)
    df$SMA10RandomSteps <- SMA(df$Steps_random,n=10)  
  }
  episode_max <- max(df$Episode)
  episode_min <- min(df$Episode)
  df$Episode_cent <- cut(df$Episode,breaks=seq(episode_min,episode_max+1,100),include.lowest = TRUE)
  return(df)
  }

# This function added an additional column to each dataframe to identify the parameters used for that experiment
add_param_label<-function(df,param,value){
  df$param_col <- value
  names(df) <- gsub('param_col',param,names(df))
  return(df)
}


# Read in data from experiment 1
df <- read.csv('Initial_experiment.csv')
df <- feat_eng(df)

########################
# Exploratory charts used to understand experiment 1

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

# Plot of the rho parameter
ggplot(df) + aes(x=`Episode`,y=`Rho`) + geom_line() + theme_light()

########################
# Charts used to represent the results of experiment 1

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
# Analysis of experiment 2; varying decay factors

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

# Analysis of the total reward within bins of 100 episodes
decfac_reward_400_500 <- decfac %>% filter(Episode_cent=='(400,500]') %>% select(df1,df2,Total_Reward) %>%
  group_by(df1,df2) %>% summarise(av_total_reward = mean(Total_Reward))

decfac_reward_300_400 <- decfac %>% filter(Episode_cent=='(300,400]') %>% select(df1,df2,Total_Reward) %>%
  group_by(df1,df2) %>% summarise(av_total_reward = mean(Total_Reward))

decfac_reward_200_300 <- decfac %>% filter(Episode_cent=='(200,300]') %>% select(df1,df2,Total_Reward) %>%
  group_by(df1,df2) %>% summarise(av_total_reward = mean(Total_Reward))

# Plot the total rewards in a heat map for each group of 100 episodes
ggplot(decfac_reward_400_500, aes(df1, df2)) + geom_tile(aes(fill = av_total_reward),colour = "white") + 
  scale_fill_gradient(low = "white",high = "steelblue",name='Mean Total Reward')
ggsave(filename="../plots/heatmap_400_500_df1df2.png", plot=last_plot(),width=2.5,height=2.5,units="in")

ggplot(decfac_reward_300_400, aes(df1, df2)) + geom_tile(aes(fill = av_total_reward),colour = "white") + 
  scale_fill_gradient(low = "white",high = "steelblue",name='Mean Total Reward')
ggsave(filename="../plots/heatmap_300_400_df1df2.png", plot=last_plot(),width=2.5,height=2.5,units="in")

ggplot(decfac_reward_200_300, aes(df1, df2)) + geom_tile(aes(fill = av_total_reward),colour = "white") + 
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

# Quantitative analysis of decay factor experiment

min(decfac_reward_400_500$av_total_reward)
max(decfac_reward_400_500$av_total_reward)


########################
# Analysis of experiment 3; varying gamma

gamma_vals <- seq(from=0.1,to=1,by=0.2)
gms <- lapply(gamma_vals,function(gamma) {
  read.csv(sprintf('experiment_gamma_%2.1f.csv',gamma)) %>%
    feat_eng() %>%
    add_param_label('gamma',gamma)
})

gm <- bind_rows(gms)
gm$gamma <- as.factor(gm$gamma)


# Line plot episode against 10 episode moving average of total reward for each gamma
ggplot(gm) + aes(x=`Episode`,y=SMA10TotalReward,colour=gamma) + geom_line() + 
  ylab("Moving average (n=10) total reward") + scale_color_discrete() + theme_light()
ggsave(filename="../plots/line_episode_reward_sma_gamma_dfs.jpg", plot=last_plot(),width=7,height=4,units="in")

# Line plot episode against total reward for each gamma
ggplot(gm) + aes(x=`Episode`,y=Total_Reward,colour=gamma) + geom_line() + 
  ylab("Total reward") + scale_color_discrete() + theme_light()


# Quantitative analysis
gamma_mean_var <- gm %>% select(Episode_cent,Total_Reward,gamma) %>% group_by(Episode_cent,gamma) %>% 
  summarise(av_total_reward = mean(Total_Reward),var_total_reward=var(Total_Reward)) %>% 
  filter(Episode_cent %in% c('[0,100]','(100,200]','(200,300]'))

dcast(gamma_mean_var, Episode_cent ~ gamma,value.var=c('av_total_reward')) %>% 
  bind_cols(dcast(gamma_mean_var,Episode_cent ~ gamma,value.var=c('var_total_reward'))) %>% View()

# Analysis of further varying gamma experiment with multiple trials
read_gamma_outputs <- function(i){
  gamma_vals <- seq(from=0.1,to=1,by=0.2)
  gms <- lapply(gamma_vals,function(gamma) {
    read.csv(sprintf('experiment_gamma_%2.1f_%d.csv',gamma,i)) %>%
      feat_eng() %>%
      add_param_label('gamma',gamma)
  })
  
  gm <- bind_rows(gms)
  gm$gamma <- as.factor(gm$gamma)
  gm$trial <- i
  gm
}

gm_trials <- lapply(seq(0,4,1),read_gamma_outputs)
gm <- bind_rows(gm_trials)

# Scatter plot of the first time each trial converges against the gamma parameter used for the trial
gamma_conv <- gm %>% group_by(gamma,trial) %>% filter(Total_Reward==max(Total_Reward)) %>% summarise(first_conv = min(Episode))
ggplot(gamma_conv) + aes(x=gamma,y=first_conv) + geom_point() + theme_light()

mean(gamma_conv$first_conv)

########################
# Analysis of initial SARSA experiment, experiment 4

srq <- read.csv('Initial_SARSA_experiment.csv')
srq <- feat_eng(srq)

# Scatter plot episode against 5 episode moving average of steps
ggplot(srq) + aes(x=`Episode`,y=SMA5Steps,color=`Outcome`) + geom_point(size=2) +
  ylab("Steps") + theme_light()
ggsave(filename="../plots/scatter_episode_steps_sma.jpg", plot=last_plot(),width=6,height=4,units="in")

# Line plot episode against 10 episode moving average of total reward
ggplot(srq) + aes(x=`Episode`,y=SMA10TotalReward) + geom_line() + 
  ylab("Moving average (n=10) total reward") + theme_light()
ggsave(filename="../plots/line_episode_reward_sma.jpg", plot=last_plot(),width=7,height=4,units="in")

# Line plot episode against 10 episode moving average of steps
ggplot(srq) + aes(x=`Episode`,y=SMA10Steps) + geom_line() + ylab("Moving Average Number of Steps in episode (n=10)") +
  scale_color_hue(labels = c("Total Steps", "Random Steps"),name="Metric") +
  theme_light()
ggsave(filename="../plots/line_episode_steps_random_sma.jpg", plot=last_plot(),width=7,height=4,units="in")

########################
# Analysis of experiment 5, varying lambda in the SARSA experiment

lambda_vals <- seq(from=0,to=1,by=0.25)
srq_lams_normsum <- lapply(lambda_vals,function(lambda) {
  read.csv(sprintf('experiment_SARSA_lambda_%3.2f_normsum_full.csv',lambda)) %>%
    feat_eng() %>%
    add_param_label('lambda',lambda)
})

srq_lam_normsum <- bind_rows(srq_lams_normsum)
srq_lam_normsum$lambda <- as.factor(srq_lam_normsum$lambda)

# Line plot episode total reward
ggplot(srq_lam_normsum) + aes(x=Episode,y=Total_Reward,colour=lambda) + geom_line() + 
  ylab("Total reward") + scale_color_discrete() + theme_light()
ggsave(filename="../plots/sarsa_lambdas_totalreward_normsum.jpg", plot=last_plot(),width=7,height=4,units="in")







