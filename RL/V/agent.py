import random
import numpy as np
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns


"""Agent class"""

class Agent():
    def __init__(self,env):        
        self.q_value = np.zeros([env.shape[0], env.shape[1], 4])   
        self.v_value = np.zeros([env.shape[0], env.shape[1]])   
        self.memory = []
    
    def learn(self,ep,a_cnt,pre_state,state,action,reward,i):
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
           
        td_target = reward + 0.9 * self.v_value[state[0],state[1]]
        td_error  = td_target - self.v_value[pre_state[0],pre_state[1]]
        self.v_value[pre_state[0],pre_state[1]] += 0.05 * td_error
        self.memorize(ep,a_cnt,action,state,reward,self.v_value[pre_state[0],pre_state[1]],self.v_value[state[0],state[1]])


    def get_action(self,ep,state):
        u=np.array([-1,0])
        r=np.array([0,1])
        l=np.array([0,-1])
        d=np.array([1,0])
        u_state = ((state[0]-1, state[1]+0))
        r_state = ((state[0]+0, state[1]+1))
        l_state = ((state[0]+0, state[1]-1))
        d_state = ((state[0]+1, state[1]+0))
        action=[u,r,l,d]
        self.epsilon = 0.5 * (1 / (ep + 1e-8))
        if np.random.random() < self.epsilon:
            a=random.choices(action)
            action=a[0] # Return as numpy type
            return action
        else:
            xx1 = self.v_value[u_state]
            xx2 = self.v_value[r_state]
            xx3 = self.v_value[l_state]
            xx4 = self.v_value[d_state]
            
            xx = np.array([xx1, xx2, xx3, xx4])

            max_x = [i for i, x in enumerate(xx) if x == max(xx)]
            max_x_index=random.choices(max_x)

            return action[max_x_index[0]]
        
    def episode_fin(self,ep):
#         sns.heatmap(self.v_value, annot=False, square=True, cmap='Greens') 
#         plt.show()
        return

    def writer(self,n):
        self.writer_csv(n)
    
    def memorize(self,ep,a_cnt,action,state,reward,stateV,n_stateV):
        # Below, long so will eventually change to loadtxt
        self.memory.append({'episode':ep,'a_cnt':a_cnt,'action':action,'s_row':state[0],'s_col':state[1],\
                        'reward':reward,'stateV':stateV,'n_stateV':n_stateV})
    
    def writer_csv(self,n):
        with open("sampleNMV-{}.csv".format(n), "w", newline="") as f:
            fieldnames = ['episode','a_cnt','action','s_row','s_col','reward','stateV','n_stateV']
            dict_writer = csv.DictWriter(f, fieldnames = fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)