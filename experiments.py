# This file contains various experiments run on the Q-learning algorithm defined in the
# FrozenQLearner class

from FrozenQLearner import FrozenQLearner, FrozenSarsaLearner
import numpy as np
from sklearn.model_selection import ParameterGrid

# Initial experiment
# fql = FrozenQLearner(episodes=500,alpha=0.1,gamma=0.7,epsilon_start=0.9,df1=0.9999,df2=0.99)
# fql.execute(log_level=30,write_file=True,file_desc='Initial experiment',norm_method='max',in_memory=False)

# Experiment with decay factors
# decay_factors = [1-0.1**n for n in range(1,7)]
# for param_set in ParameterGrid({'df1':decay_factors,'df2':decay_factors}):
#     fql = FrozenQLearner(episodes=500, alpha=0.1, gamma=0.7, epsilon_start=0.9, df1=param_set['df1'], df2=param_set['df2'])
#     fql.execute(log_level=30, write_file=True, file_desc='experiment_dfs_%7.6f_%7.6f' % (param_set['df1'],param_set['df2']), in_memory=False)

# Experiment in varying gamma
# for gamma_test in np.arange(0.1,1.0,0.2):
#     fql = FrozenQLearner(episodes=500, alpha=0.1, gamma=gamma_test, epsilon_start=0.9, df1=0.9999, df2=0.99)
#     fql.execute(log_level=30, write_file=True, file_desc='experiment_gamma_%2.1f' % gamma_test, in_memory=False)

# Experiment in varying alpha
# for alpha_test in np.arange(0.1,0.6,0.2):
#     fql = FrozenQLearner(episodes=500, alpha=alpha_test, gamma=0.7, epsilon_start=0.9, df1=0.9999, df2=0.99)
#     fql.execute(log_level=30, write_file=True, file_desc='experiment_alpha_%2.1f' % alpha_test, in_memory=False)


# Initial SARSA experiment, set lambda to 0 for Q-learning
fsl = FrozenSarsaLearner(episodes=500,alpha=0.1,gamma=0.7,td_lambda=0)
fsl.execute(log_level=0,write_file=True,file_desc='Initial SARSA experiment')
