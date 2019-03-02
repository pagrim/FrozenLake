# This file contains various experiments run on the Q-learning algorithm defined in the
# FrozenQLearner class

from FrozenQLearner import FrozenQLearner
import numpy as np


# Initial experiment
# fql = FrozenQLearner(episodes=500,alpha=0.1,gamma=0.7,epsilon_start=0.9,df1=0.9999,df2=0.99)
# fql.execute(log_level=0,write_file=True,file_desc='Initial experiment',in_memory=False)

# Experiment in varying gamma
for gamma_test in np.arange(0.1,1.0,0.2):
    fql = FrozenQLearner(episodes=500, alpha=0.1, gamma=gamma_test, epsilon_start=0.9, df1=0.9999, df2=0.99)
    fql.execute(log_level=30, write_file=True, file_desc='experiment_gamma_%2.1f' % gamma_test, in_memory=False)



