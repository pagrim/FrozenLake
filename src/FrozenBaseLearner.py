import numpy as np
import logging
import io
from gym.envs.registration import register
from gym import make

# Create an id for a new Frozen Lake environment based on a 4x4 map which is not slippery
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4', 'is_slippery': False}
)

# Instantiate logger
logger = logging.getLogger('frozen_logger')
logger.setLevel(logger.info)
logger.addHandler(logging.StreamHandler())

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

    def rdm_opt_act(self, state):
        """ Choose at random an action from the set of actions which have equal values in the Q matrix. Otherwise the
        learner will deterministically select an action, favouring some states. Excludes NaN values from being selected.
        :param state: int
            state from which to choose action
        :return action: int
            integer representation of chosen action
        """
        poss_Q = self.Q[state, :]
        max_inds = [i for i, o_a in enumerate(poss_Q) if o_a == np.nanmax(poss_Q)]
        logger.debug('Indices of optimal actions %s', max_inds)
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
        logger.debug('Selecting from Q values %s', poss_Q)
        action = np.random.randint(self.numA)
        while np.isnan(poss_Q[action]):
            action = np.random.randint(self.numA)
        return action

    def update_rho(self, reward):
        self.rho = self.rho + self.alpha*(reward - self.rho)

    # Write the results of each episode to file
    @staticmethod
    def open_file(in_memory, file_desc, header):
        if not in_memory:
            outfile = open('outputs/%s.csv' % file_desc.replace(' ', '_'), 'w')
        else:
            outfile = io.StringIO()
        outfile.write('%s\n' % header)
        return outfile





