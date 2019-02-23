import numpy as np
import logging
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import datetime as dt

class FrozenQLearner:

    def __init__(self,episodes,alpha,gamma,epsilon_start,df1,df2,is_slippery=False):

        # Initialise FL environment
        self.FLenv = FrozenLakeEnv(is_slippery=is_slippery)
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
        self.epsilon = epsilon_start
        self.episodes = episodes

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
        self.Q = np.array([[0 if not np.isnan(el) else np.nan for el in row] for row in self.R])

    def update_Q(self,state_1,action,reward,state_2):
        learned_value = self.R[state_1,action] + self.gamma*self.Q[state_2,np.nanargmax(self.Q[state_2,:])]
        self.Q[state_1,action] = (1-self.alpha)*self.Q[state_1,action]+self.alpha*(learned_value)

    @staticmethod
    def rdm_opt_act(Qvals):
        max_inds = [i for i, o_a in enumerate(Qvals) if o_a == np.nanmax(Qvals)]
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
        poss_Q = self.Q[state, :]
        if np.random.random() > self.epsilon:
            logging.debug('Selecting random action')
            poss_R = self.R[state,:]
            action = self.rdm_poss_act(poss_R,state)
        else:
            logging.debug('Selecting from Q values %s',poss_Q)
            action = self.rdm_opt_act(poss_Q)
        return action

    def update_epsilon(self,new_epsilon):
        self.epsilon = new_epsilon


    def execute(self,log_level,write_file,file_desc):

        logging.basicConfig(level=log_level)

        episode = 0
        state = self.FLenv.reset()
        self.init_R(val_goal=100,val_other=0,wall_moves=False)
        logging.debug('Reward matrix %s',self.R)
        self.init_Q()

        def episode_metrics():
            return '%d,%d,%4.2f,%s' % (episode,ep_steps,ep_total_reward,ep_outcome)

        def metric_headers():
           return 'Episode, Steps, Total_Reward, Outcome'

        # Write the results of each episode to file
        if write_file:
            ts = dt.datetime.now()
            outfile = open('outputs/%d%d%d_%d_%d_%s.csv' %
                           (ts.year, ts.month, ts.day, ts.hour, ts.minute,file_desc.replace(' ','_')),'w')
            outfile.write(metric_headers())

        # Start Q-learning
        while episode < self.episodes:
            episode_complete = False
            step = 0
            ep_total_reward = 0
            while not episode_complete:

                if log_level >= 0:
                    self.FLenv.render()
                action = self.select_action(state)
                logging.info('Action chosen: %d',action)
                logging.debug('Action feasible: %s', not np.isnan(self.R[state,action]))
                state_new, _, episode_complete, _ = self.FLenv.step(action)

                logging.info('New state is %d',state_new)
                reward = self.R[state, action]
                logging.info('Reward: %d',reward)
                self.update_Q(state, action, reward, state_new)
                logging.info('Q matrix updated: %s',self.Q)
                state, state_new = state_new, None
                ep_total_reward += reward
                logging.info('*** Next iteration *** Step %d, Episode %d',step, episode)
                step += 1

            ep_outcome = self.map[self.from_state(state)]
            state = self.FLenv.reset()
            state_new = None

            # TODO change method for update epsilon (extract np.random.random from select_action)

            # Calculate and report metrics for the episode
            ep_steps = step - 1 # Account for the +=1 in each while loop
            ep_met = episode_metrics()
            logging.info('%s\n%s\n%s\n%s', '*' * 30,metric_headers(),ep_met, '*' * 30)
            if write_file:
                outfile.write(ep_met)

            episode += 1

        if write_file:
            outfile.close()




