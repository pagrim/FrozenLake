from src.FrozenBaseLearner import FrozenLearner
import logging
import numpy as np

logger = logging.getLogger('frozen_logger')


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
        super(FrozenQLearner, self).__init__(episodes, alpha, gamma)
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
            logger.debug('Selecting random action')
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
        logger.info('%s\nRunning Q-learning Experiment\n alpha=%3.2f,gamma=%3.2f,epsilon_start=%3.2f,''df1=%3.2f,'
                    'df2=%3.2f \n%s', '*' * 30, self.alpha, self.gamma, self.epsilon, self.df1, self.df2, '*' * 30)
        logger.debug('Reward matrix %s', self.R)
        self.init_Q()

        # Define the data to record for each episode
        def episode_metrics():
            return '%d,%d,%4.2f,%s,%d,%4.2f' % (
                episode, ep_steps, ep_total_reward, ep_outcome, ep_steps_random, ep_epsilon_start)

        # Define the headers for the recorded data
        def metric_headers():
            return 'Episode,Steps,Total_Reward,Outcome,Steps_random,Epsilon_start,Rho'

        if write_file:
            outfile = self.open_file(in_memory, file_desc, metric_headers())

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
                logger.info('Action chosen: %d', action)
                logger.debug('Action feasible: %s', not np.isnan(self.R[state, action]))
                # Implement the action in the Frozen Lake environment
                state_new, _, episode_complete, _ = self.FLenv.step(action)

                logger.info('New state is %d', state_new)
                reward = self.R[state, action]
                logger.info('Reward: %d', reward)
                self.update_Q(state, action, reward, state_new)
                self.normalise_Q(norm_method)
                logger.info('Q matrix updated: %s', self.Q)
                # Update our learning metric
                self.update_rho(reward)
                # Add the reward for this step to the cumulative reward for the episode
                ep_total_reward += self.Q[state, action]
                state, state_new = state_new, None
                self.update_epsilon(random_value)

                logger.info('*** Completed Step %d of Episode %d ***', step, episode)
                step += 1

            ep_outcome = self.map[self.from_state(state)]
            state = self.FLenv.reset()
            state_new = None

            # Calculate and report metrics for the episode
            ep_steps = step
            ep_met = episode_metrics()
            logger.info('\nEpisode Complete\n%s\n%s\n%s\n%s', '*' * 30, metric_headers(), ep_met, '*' * 30)
            if write_file:
                outfile.write('%s\n' % ep_met)

            episode += 1

        if write_file:
            if in_memory:
                return outfile
            outfile.close()