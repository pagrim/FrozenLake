# This file contains various experiments run on the Q-learning algorithm defined in the
# FrozenQLearner class

from FrozenQLearner import FrozenQLearner
import pandas as pd

#fql = FrozenQLearner(episodes=200,alpha=0.5,gamma=0.6,epsilon_start=0.9,df1=0.9999,df2=0.99)

fql = FrozenQLearner(episodes=300,alpha=0.5,gamma=0.6,epsilon_start=0.9,df1=1,df2=1)

output = fql.execute(log_level=0,write_file=True,file_desc='Initial test',in_memory=True)
output.seek(0)

print(pd.read_csv(output).head(5))


