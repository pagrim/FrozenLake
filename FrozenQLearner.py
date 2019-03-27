import numpy as np
import logging
import io
from gym.envs.registration import register
from gym import make

# Create an id for a new Frozen Lake environment based on a 4x4 map which is not slippery
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False}
)

class FrozenLearner:
    """ Base class for Q-matrix based reinforcement learning algorithms on the Frozen Lake problem"""

    def __init__(self, episodes, alpha, gamma):
        """
        :param episodes: int
            Maximum number of episodes for learner
        :param alpha: float
            Learning rate
        :param gamma: float
            Discount applied to future states
        """

        # Initialise FL environment
        self.FLenv = make('FrozenLakeNotSlippery-v0')
        self.map = map = self.FLenv.desc

        # Check the map row sizes are consistent
        row_lens = [len(row) for row in map]
        assert (min(row_lens) == max(row_lens) & len(map) == row_lens[0]), "Inconsistent row sizes"

        # Set the number of states and actions
        self.mapLen = len(map)
        self.numS = numS = len(map) * len(map[0])
        self.numA = numA = 4

        # Initialise empty R and Q matrices
        self.R = np.empty((numS, numA)) * np.nan
        self.Q = np.empty((numS, numA)) * np.nan

        # Initialise other parameters
        self.alpha = alpha
        self.gamma = gamma
        self.episodes = episodes
        self.rho = 0

    # Retrieve the state value from a row or col on the map
    def get_state(self, map_row, map_col):
        return map_row * self.mapLen + map_col

    # Convert a row or column on the map to a state value
    def from_state(self, state):
        return (state // self.mapLen, state % self.mapLen)

    def evaluate_action(self, map_row, map_col, a):
        """
        :param map_row: int
            row of the Frozen Lake map
        :param map_col: int
            col of the Frozen Lake map
        :param a: int
            an action value
        :return map_row,map_col: int, int
            resulting row and column in the map
        """
        if a == 0:  # left
            map_col = max(map_col - 1, 0)
        elif a == 1:  # down
            map_row = min(map_row + 1, self.mapLen - 1)
        elif a == 2:  # right
            map_col = min(map_col + 1, self.mapLen - 1)
        elif a == 3:  # up
            map_row = max(map_row - 1, 0)
        return map_row, map_col

    # Identify whether a move results in a wall move, i.e. the agent staying in place
    def is_wall_move(self, row, col, a):
        return self.evaluate_action(row, col, a) == (row, col)

    # Initialise the R matrix including NaN values for wall moves
    def init_R(self, val_goal, val_other, wall_moves):
        if wall_moves:
            self.R.fill(0)
        for rowS in range(self.numS):
            for colA in range(self.numA):
                map_row, map_col = self.from_state(rowS)
                if not wall_moves:
                    if self.is_wall_move(map_row, map_col, colA):
                        self.R[rowS, colA] = np.nan
                    elif self.map[self.evaluate_action(map_row, map_col, colA)] == b'G':
                        self.R[rowS, colA] = val_goal
                    else:
                        self.R[rowS, colA] = val_other

    # Initialise the Q matrix with 0 in place of NaN values
    def init_Q(self):
        assert (self.R is not None), "Missing R matrix"
        self.Q = np.array([[0 if not np.isnan(el) else np.nan for el in row] for row in self.R])

    def normalise_Q(self, norm_type):
        """ Adjusts the Q matrix for fair comparison between parameter values. Otherwise, for example the total reward
        of an episode can depend on the discount parameter
        :param norm_type: str
             either 'sum' or 'max'
        :return: numpy array
            normalised Q matrix
        """
        if norm_type == 'sum':
            if np.nansum(self.Q) > 0:
                self.Q = self.Q / np.nansum(self.Q)
        elif norm_type == 'max':
            if np.nanmax(self.Q) > 0:
                self.Q = self.Q / np.nanmax(self.Q)

    def rdm_opt_act(self,state):
        """ Choose at random an action from the set of actions which have equal values in the Q matrix. Otherwise the
        learner will deterministically select an action, favouring some states. Excludes NaN values from being selected.
        :param state: int
            state from which to choose action
        :return action: int
            integer representation of chosen action
        """
        poss_Q = self.Q[state, :]
        max_inds = [i for i, o_a in enumerate(poss_Q) if o_a == np.nanmax(poss_Q)]
        logging.debug('Indices of optimal actions %s',max_inds)
        if len(max_inds) > 1:
            action = max_inds[np.random.randint(len(max_inds))]
        else:
            action = np.nanargmax(poss_Q)
        return action

    def rdm_poss_act(self, state):
        """ Choose an action at random excluding NaN values
        :param state: int
            state from which to choose action
        :return action: int
            integer representation of chosen action
        """
        poss_Q = self.Q[state, :]
        logging.debug('Selecting from Q values %s', poss_Q)
        action = np.random.randint(self.numA)
        while np.isnan(poss_Q[action]):
            action = np.random.randint(self.numA)
        return action

    def update_rho(self, reward):
        self.rho = self.rho + self.alpha*(reward - self.rho)

    # Write the results of each episode to file
    @staticmethod
    def open_file(in_memory,file_desc,header):
        if not in_memory:
            outfile = open('outputs/%s.csv' % file_desc.replace(' ', '_'), 'w')
        else:
            outfile = io.StringIO()
        outfile.write('%s\n' % header)
        return outfile


class FrozenQLearner(FrozenLearner):
    """Implementation of the Q-learning method described by Watkins (1989) for the Frozen Lake problem"""

    def __init__(self, episodes, alpha, gamma, epsilon_start, df1, df2):
        """
        :param episodes: int
            maximum number of episodes
        :param alpha: int
            learning rate
        :param gamma: float
            discount rate for future rewards and Q values
        :param epsilon_start: float
            initial value for epsilon, which is reduced by the decay factors each episode
        :param df1:
            first decay factor
        :param df2:
            second decay factor
        """
        super(FrozenQLearner,self).__init__(episodes,alpha,gamma)
        self.epsilon = epsilon_start
        self.df1 = df1
        self.df2 = df2

    def update_Q(self, state_1, action, reward, state_2):
        """ Updates the Q matrix based on the Q-learning update rule
        :param state_1: int
            the state at time t
        :param action: int
            the action at time t
        :param reward: float
            the reward at time t
        :param state_2: int
            the state at time t+1
        """
        learned_value = reward + self.gamma * self.Q[state_2, np.nanargmax(self.Q[state_2, :])]
        self.Q[state_1, action] = (1 - self.alpha) * self.Q[state_1, action] + self.alpha * (learned_value)

    # Multiplies epsilon by the relevant decay factor
    def update_epsilon(self, random_value):
        if random_value > self.epsilon:
            self.epsilon *= self.df1
        else:
            self.epsilon *= self.df2

    def select_action(self, state, random_value):
        """ Selects either a random action or argmax Q value depending on the values of epsilon and random_value
        :param state: int
            current state
        :param random_value: float
            a randomly generated number for the current step
        :return action, is_random: int, bool
             action selected, whether this was random
        """
        is_random = random_value < self.epsilon
        if is_random:
            logging.debug('Selecting random action')
            action = self.rdm_poss_act(state)
        else:
            action = self.rdm_opt_act(state)
        return action, is_random

    def execute(self, log_level, write_file, file_desc, norm_method='max', in_memory=False):
        """ Main method to run the Q-learning algorithm
        :param log_level: int
            Number between 0 and 50; 0 logs everything, see documentation for logging package
        :param write_file: bool
            Indicates whether to create an output file
        :param file_desc: str
            Description for the output file
        :param norm_method: str
            Method of normalisation
        :param in_memory: bool
            Should the ouput file be returned in memmory
        """

        logging.basicConfig(level=log_level)

        episode = 0
        state = self.FLenv.reset()
        self.init_R(val_goal=100, val_other=0, wall_moves=False)
        logging.info('%s\nRunning Q-learning Experiment\n alpha=%3.2f,gamma=%3.2f,epsilon_start=%3.2f,''df1=%3.2f,'
                     'df2=%3.2f \n%s','*' * 30, self.alpha, self.gamma,self.epsilon,self.df1,self.df2, '*' * 30)
        logging.debug('Reward matrix %s', self.R)
        self.init_Q()

        # Define the data to record for each episode
        def episode_metrics():
            return '%d,%d,%4.2f,%s,%d,%4.2f,%4.2f' % (
            episode, ep_steps, ep_total_reward, ep_outcome, ep_steps_random, ep_epsilon_start, self.rho)

        # Define the headers for the recorded data
        def metric_headers():
            return 'Episode,Steps,Total_Reward,Outcome,Steps_random,Epsilon_start,Rho'

        if write_file:
            outfile = self.open_file(in_memory,file_desc,metric_headers())

        # Start Q-learning
        while episode < self.episodes:
            episode_complete = False
            step = 0
            ep_total_reward = 0
            ep_epsilon_start = self.epsilon
            ep_steps_random = 0

            while not episode_complete:

                random_value = np.random.random()
                if log_level <= 20:
                    self.FLenv.render()
                action, step_random = self.select_action(state, random_value)
                # Record whether the step was random
                ep_steps_random += int(step_random)
                logging.info('Action chosen: %d', action)
                logging.debug('Action feasible: %s', not np.isnan(self.R[state, action]))
                # Implement the action in the Frozen Lake environment
                state_new, _, episode_complete, _ = self.FLenv.step(action)

                logging.info('New state is %d', state_new)
                reward = self.R[state, action]
                logging.info('Reward: %d', reward)
                self.update_Q(state, action, reward, state_new)
                self.normalise_Q(norm_method)
                logging.info('Q matrix updated: %s', self.Q)
                # Update our learning metric
                self.update_rho(reward)
                # Add the reward for this step to the cumulative reward for the episode
                ep_total_reward += self.Q[state, action]
                state, state_new = state_new, None
                self.update_epsilon(random_value)

                logging.info('*** Completed Step %d of Episode %d ***', step, episode)
                step += 1

            ep_outcome = self.map[self.from_state(state)]
            state = self.FLenv.reset()
            state_new = None

            # Calculate and report metrics for the episode
            ep_steps = step
            ep_met = episode_metrics()
            logging.info('\nEpisode Complete\n%s\n%s\n%s\n%s', '*' * 30, metric_headers(), ep_met, '*' * 30)
            if write_file:
                outfile.write('%s\n' % ep_met)

            episode += 1

        if write_file:
            if in_memory:
                return outfile
            outfile.close()


class FrozenSarsaLearner(FrozenLearner):
    """Implementation of the SARSA algorithm proposed by Rummery and Niranjan (1994)"""

    def __init__(self, episodes, alpha, gamma, td_lambda):
        """
        :param episodes: int
            maximum number of epsiodes for SARSA
        :param alpha: float
            learning rate
        :param gamma: float
            discount for future rewards
        :param td_lambda: float
            discount parameter for eligibility traces, lambda=0 => ~ Q-learning and lambda =1 => ~ Monte Carlo
        """
        super(FrozenSarsaLearner,self).__init__(episodes,alpha,gamma)
        self.td_lambda = td_lambda

    # Initialise the E matrix of eligibility traces
    def init_E(self):
        self.E = np.zeros((self.numS,self.numA))

    # Select an action based on the method parameter passed at runtime
    def select_action(self, state, method):
        poss_Q = self.Q[state, :]
        logging.debug('Selecting from Q values %s', poss_Q)
        if method=='argmax_rand':
            return self.rdm_opt_act(state)
        elif method=='argmax_true':
            return np.nanargmax(poss_Q)
        else:
            raise ValueError('Unknown method')

    def update_E(self, state, action):
        self.E *= self.gamma * self.td_lambda
        self.E[state,action] = 1

    def update_Q(self, learned_value):
        self.Q += (self.alpha * learned_value * self.E)

    def learned_value(self,state,action,state_new,action_new):
        return self.R[state, action] + self.gamma * (self.Q[state_new, action_new] - self.Q[state, action])

    def execute(self, log_level, write_file, file_desc, norm_method='max', select_method='non-random', in_memory=False):
        """ Main method to run the SARSA algorithm
        :param log_level: int
            Number between 0 and 50; 0 logs everything, see documentation for logging package
        :param write_file: bool
            Indicates whether to create an output file
        :param file_desc: str
            Description for the output file
        :param norm_method: str
            Method of normalisation
        :param select_method: str
            either 'argmax_rand' or 'argmax_true', method for choosing between equal values in the Q matrix
        :param in_memory: bool
            Should the ouput file be returned in memmory
        """

        logging.basicConfig(level=log_level)

        episode = 0
        state = self.FLenv.reset()
        self.init_R(val_goal=100, val_other=0, wall_moves=False)
        logging.info('%s\nRunning SARSA Experiment\n alpha=%3.2f,gamma=%3.2f,lambda=%3.2f \n%s',
                     '*' * 30, self.alpha, self.gamma,self.td_lambda, '*' * 30)
        logging.debug('Reward matrix %s', self.R)
        self.init_Q()
        self.init_E()

        # Define the data to record for each episode
        def episode_metrics():
            return '%d,%d,%4.2f,%s' % (episode, ep_steps, ep_total_reward, ep_outcome)

        # Define the headers for the recorded data
        def metric_headers():
            return 'Episode,Steps,Total_Reward,Outcome'

        # Write the results of each episode to file
        if write_file:
            outfile = self.open_file(in_memory, file_desc, metric_headers())

        # Start SARSA
        while episode < self.episodes:
            episode_complete = False
            step = 0
            ep_total_reward = 0
            action = self.rdm_opt_act(state)
            self.init_E()

            while not episode_complete:

                if log_level <= 20:
                    self.FLenv.render()
                reward = self.R[state, action]
                logging.debug('State %d,action %d before reward',state,action)
                logging.info('Reward: %d', reward)
                state_new, _, episode_complete, _ = self.FLenv.step(action)
                logging.info('New state is %d', state_new)
                action_new = self.select_action(state_new,select_method)
                logging.info('New action chosen: %d', action_new)
                logging.debug('New action feasible: %s', not np.isnan(self.R[state_new, action_new]))
                # Update eligibility trace matrix from current state
                self.update_E(state,action)
                logging.info('E matrix updated: %s',self.E)
                # Identify the learned value according to SARSA update rule, sometimes labelled as delta
                delta = self.learned_value(state,action,state_new,action_new)
                logging.debug('Learned value: %4.2f',delta)
                # Update the Q matrix based on learned value
                self.update_Q(delta)
                self.normalise_Q(norm_method)
                logging.info('Q matrix updated: %s', self.Q)

                ep_total_reward += self.Q[state, action]

                state, state_new = state_new, None
                action, action_new = action_new, None

                logging.info('*** Completed Step %d of Episode %d ***', step, episode)
                step += 1

            ep_outcome = self.map[self.from_state(state)]
            state = self.FLenv.reset()
            state_new = None

            # Calculate and report metrics for the episode
            ep_steps = step  # N.B steps are numbered from 0 but +=1 in loop accounts for this
            ep_met = episode_metrics()
            logging.info('\nEpisode Complete\n%s\n%s\n%s\n%s', '*' * 30, metric_headers(), ep_met, '*' * 30)
            if write_file:
                outfile.write('%s\n' % ep_met)

            episode += 1

        if write_file:
            if in_memory:
                return outfile
            outfile.close()
