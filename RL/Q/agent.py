import random
import numpy as np
import csv

"""Agent class"""

class Agent():
    def __init__(self,env):
        self.q_value = np.zeros([env.shape[0], env.shape[1], 4])
        self.memory = []
        # random.seed(14)
    
    def learn(self,ep,a_cnt,pre_state,state,action,reward):
        if action[0]==-1:
            a=0
            b=3
        elif action[1]==1:
            a=1
            b=2
        elif action[1]==-1:
            a=2
            b=1
        elif action[0]==1:
            a=3
            b=0

        self.q_value[pre_state[0],pre_state[1],a] += 0.05 * (reward + 0.9 * np.max(self.q_value[state[0], state[1]]) - self.q_value[pre_state[0],pre_state[1],a])
        next_q = np.max(self.q_value[state[0], state[1]])
        self.memorize(ep,a_cnt,action,state,reward,self.q_value[pre_state[0],pre_state[1],a], 0.9*next_q)

    def get_action(self,ep,state):
        u=np.array([-1,0])
        r=np.array([0,1])
        l=np.array([0,-1])
        d=np.array([1,0])
        action=[u,r,l,d]

        self.epsilon = 0.5 * (1 / (ep + 1e-8))
        if np.random.random() < self.epsilon:
            a=random.choices(action)
            action=a[0] # Return as numpy type
            return action
        else:
            xx = np.array([self.q_value[state[0], state[1], 0], self.q_value[state[0], state[1], 1], self.q_value[state[0], state[1], 2], self.q_value[state[0], state[1], 3]])
            max_x = [i for i, x in enumerate(xx) if x == max(xx)]
            max_x_index=random.choices(max_x)
            return action[max_x_index[0]]
        
    def episode_fin(self,ep):
        return

    def writer(self,n):
        self.writer_csv(n)
    
    def memorize(self,ep,a_cnt,action,state,reward,stateQ,n_stateQ):
        # Below, it's long so will eventually change to loadtxt
        self.memory.append({'episode':ep,'a_cnt':a_cnt,'action':action,'s_row':state[0],'s_col':state[1],\
                        'reward':reward,'stateQ':stateQ,'n_stateQ':n_stateQ})
    
    def writer_csv(self,n):
        with open("sampleNMQ_200-{}.csv".format(n), "w", newline="") as f:
            fieldnames = ['episode','a_cnt','action','s_row','s_col','reward','stateQ','n_stateQ']
            dict_writer = csv.DictWriter(f, fieldnames = fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)