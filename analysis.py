# This file contains various experiments run on the Q-learning algorithm defined in the
# FrozenQLearner class

from FrozenQLearner import FrozenQLearner
import pandas as pd

#fql = FrozenQLearner(episodes=200,alpha=0.5,gamma=0.6,epsilon_start=0.9,df1=0.9999,df2=0.99)

fql = FrozenQLearner(episodes=500,alpha=0.1,gamma=0.7,epsilon_start=0.9,df1=0.9999,df2=0.99)

fql.execute(log_level=0,write_file=True,file_desc='Initial experiment',in_memory=False)

#output.seek(0)

#print(pd.read_csv(output).head(5))


