from src.FrozenBaseLearner import FrozenLearner
import logging
import numpy as np

logger = logging.getLogger('frozen_logger')


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
        super(FrozenSarsaLearner, self).__init__(episodes, alpha, gamma)
        self.td_lambda = td_lambda

    # Initialise the E matrix of eligibility traces
    def init_E(self):
        self.E = np.zeros((self.numS, self.numA))

    # Select an action based on the method parameter passed at runtime
    def select_action(self, state, method):
        poss_Q = self.Q[state, :]
        logger.debug('Selecting from Q values %s', poss_Q)
        if method == 'argmax_rand':
            return self.rdm_opt_act(state)
        elif method == 'argmax_true':
            return np.nanargmax(poss_Q)
        else:
            raise ValueError('Unknown method')

    def update_E(self, state, action):
        self.E *= self.gamma * self.td_lambda
        self.E[state, action] = 1

    def update_Q(self, learned_value):
        self.Q += (self.alpha * learned_value * self.E)

    def learned_value(self, state, action, state_new, action_new):
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
        logger.info('%s\nRunning SARSA Experiment\n alpha=%3.2f,gamma=%3.2f,lambda=%3.2f \n%s',
                     '*' * 30, self.alpha, self.gamma, self.td_lambda, '*' * 30)
        logger.debug('Reward matrix %s', self.R)
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
                logger.debug('State %d,action %d before reward', state, action)
                logger.info('Reward: %d', reward)
                state_new, _, episode_complete, _ = self.FLenv.step(action)
                logger.info('New state is %d', state_new)
                action_new = self.select_action(state_new, select_method)
                logger.info('New action chosen: %d', action_new)
                logger.debug('New action feasible: %s', not np.isnan(self.R[state_new, action_new]))
                # Update eligibility trace matrix from current state
                self.update_E(state, action)
                logger.info('E matrix updated: %s', self.E)
                # Identify the learned value according to SARSA update rule, sometimes labelled as delta
                delta = self.learned_value(state, action, state_new, action_new)
                logger.debug('Learned value: %4.2f', delta)
                # Update the Q matrix based on learned value
                self.update_Q(delta)
                self.normalise_Q(norm_method)
                logger.info('Q matrix updated: %s', self.Q)

                ep_total_reward += self.Q[state, action]

                state, state_new = state_new, None
                action, action_new = action_new, None

                logger.info('*** Completed Step %d of Episode %d ***', step, episode)
                step += 1

            ep_outcome = self.map[self.from_state(state)]
            state = self.FLenv.reset()
            state_new = None

            # Calculate and report metrics for the episode
            ep_steps = step  # N.B steps are numbered from 0 but +=1 in loop accounts for this
            ep_met = episode_metrics()
            logger.info('\nEpisode Complete\n%s\n%s\n%s\n%s', '*' * 30, metric_headers(), ep_met, '*' * 30)
            if write_file:
                outfile.write('%s\n' % ep_met)

            episode += 1

        if write_file:
            if in_memory:
                return outfile
            outfile.close()
