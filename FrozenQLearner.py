import numpy as np
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

class FrozenQLearner:

    def __init__(self,alpha,gamma,epsilon):

        # Initialise FL environment
        self.FLenv = FrozenLakeEnv(is_slippery=False)
        self.map = map = self.FLenv.desc

        # Check the map row sizes are consistent
        row_lens = [len(row) for row in map]
        assert(min(row_lens)==max(row_lens) & len(map)==row_lens[0]), "Inconsistent row sizes"

        #Get the number of states
        self.mapLen = len(map)
        self.numS = numS = len(map)*len(map[0])
        self.numA = numA = 4

        # Initialise empty R and Q matrices
        self.R = np.empty((numS,numA))*np.nan
        self.Q = np.empty((numS,numA))*np.nan

        # Initialise parameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, map_row, map_col):
        return map_row * self.mapLen + map_col

    def from_state(self, state):
        return (state // self.mapLen ,state % self.mapLen)

    def evaluate_action(self, map_row, map_col, a):
        if a == 0:  # left
            map_col = max(map_col - 1, 0)
        elif a == 1:  # down
            map_row = min(map_row + 1, self.mapLen - 1)
        elif a == 2:  # right
            map_col = min(map_col + 1, self.mapLen - 1)
        elif a == 3:  # up
            map_row = max(map_row - 1, 0)
        return (map_row, map_col)

    def is_wall_move(self,row,col,a):
        return self.evaluate_action(row,col,a)==(row,col)

    def init_R(self,val_goal,val_other,wall_moves):
        if wall_moves:
            self.R.fill(0)
        for rowS in range(self.numS):
            for colA in range(self.numA):
                map_row, map_col = self.from_state(rowS)
                if not wall_moves:
                    if self.is_wall_move(map_row,map_col,colA):
                        self.R[rowS,colA] = np.nan
                    elif self.map[self.evaluate_action(map_row,map_col,colA)] == b'G':
                        self.R[rowS, colA] = val_goal
                    else:
                        self.R[rowS,colA] = val_other

    def init_Q(self):
        assert(self.R is not None), "Missing R matrix"
        self.Q = np.zeros((self.numS,self.numA))

    def update_Q(self,state_1,action,reward,state_2):
        learned_value = self.R[state_1,action] + self.gamma*self.Q[state_2,np.argmax(self.Q[state_2,:])]
        self.Q[state_1,action] = (1-self.alpha)*self.Q[state_1,action]+self.alpha*(learned_value)

    @staticmethod
    def rdm_opt_act(Qvals):
        max_inds = [i for i, o_a in enumerate(Qvals) if o_a == np.nanmax(Qvals)]
        # Handle multiple optimal Q values in Q matrix by random selection of the optimal actions
        if len(max_inds) > 1:
            action = max_inds[np.random.randint(len(max_inds))]
        else:
            action = np.nanargmax(Qvals)
        return action

    def rdm_poss_act(self,poss_R,state):
        action = np.random.randint(self.numA)
        while np.isnan(self.R[state, action]):
            action = np.random.randint(self.numA)
        return action

    def select_action(self,state):
        if np.random.random() > self.epsilon:
            print('Selecting random action')
            poss_R = self.R[state,:]
            self.rdm_poss_act(poss_R,state)
        else:
            poss_Q = self.Q[state, :]
            print('Selecting from Q values %s' % poss_Q)
            action = self.rdm_opt_act(poss_Q)
            ### Need something to ensure a random optimal Q value is actually possible
        return action

    def update_epsilon(self,new_epsilon):
        self.epsilon = new_epsilon

if __name__ =='__main__':
    testLearner = FrozenQLearner(0.5,0.5,0.6)
    testLearner.init_R(100,0,False)
    print(testLearner.R)
    testLearner.init_Q()

    NUM_ITERATIONS = 30

    # Reset the environment and return the initial state
    state = testLearner.FLenv.reset()
    print('Initial state is %d' % state)

    for iter in range(NUM_ITERATIONS):
        # Select action, either random or maximum Q value
        action = testLearner.select_action(state)
        print('Action selected is %d' % action)
        # Take the action in the environment and observe new state
        state_new, _, done, _ = testLearner.FLenv.step(action)
        print('New state is %d' % state_new)
        # Identify the reward of this action
        reward = testLearner.R[state,action]
        # Update Q based on the reward of the action
        testLearner.update_Q(state,action,reward,state_new)
        print('Q updated:')
        print(testLearner.Q)
        # Update states
        state, state_new = state_new, None
        print('*** Next iteration ***')