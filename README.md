# Frozen Lake
The code in this repository aims to solve the Frozen Lake problem, one of the problems in AI gym, using Q-learning and SARSA

## Algortithms
The `FrozenQLearner.py` file contains a base `FrozenLearner` class and two subclasses `FrozenQLearner` and `FrozenSarsaLearner`. These are called by the `experiments.py` file.

## Experiments
The `experiments.py` file contains the details of the experiments run using the two algorithms. If the `output_file` parameter of the methods is set to true a CSV file summarising each session will be written to the `outputs` directory. The experiments call the `FrozenLearner` subclasses. 

## Analysis
The `analysis.R` file was used to analyse the output of experiments using charts and some quantitative analysis. The charts are saved to the `plots` directory in some cases
