import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns


class BG_Network():

    def __init__(self, maze_shape, i):
        """Build network according to the environment"""
        self.action_num = 4 # Number of action choices
        self.shape      = maze_shape.shape # Shape of the network
        self.maze       = maze_shape

        """Cell settings"""
        self.x     = np.zeros(self.shape) # Cerebral cortex (input)
        self.stn   = np.zeros(self.shape) # Subthalamic nucleus (D5 receptor)
        self.d1    = np.zeros(self.shape) # D1 receptor
        self.d2    = np.zeros(self.shape) # D2 receptor
        self.gpe   = np.zeros(self.shape) # External segment of the globus pallidus
        self.snr   = np.zeros(self.shape + tuple([self.action_num + 1])) # Internal segment of the globus pallidus
        self.xs    = np.zeros(self.shape + tuple([self.action_num + 1])) # Cerebral cortex (re-input)
        self.strio = np.zeros(self.shape + tuple([self.action_num + 1])) # Striosome
        self.da    = np.zeros(self.shape) # Dopaminergic cells

        """Input"""
        self.input = 1 # Input

        """Weight parameters"""
        self.wstn   = np.ones(self.shape) # Cerebral cortex - Subthalamic nucleus (D5 receptor)
        self.wd1    = np.ones(self.shape) # Cerebral cortex - D1 receptor
        self.wd2    = np.ones(self.shape) # Cerebral cortex - D2 receptor
        self.wstrio = np.ones(self.shape)*5 # Cerebral cortex - Striosome

        """Size parameters"""
        gl_stn_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] #05/23 14:06 @ lab pc 05/24 12:48 @ mine
        gl_gpe_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1] # Published in PICO
        lc_stn_list = [1, 5.5, 7, 7.75, 8.2, 8.5, 8.714, 8.875, 9]  # intra d1=5, gpe=5 changes required!!!!!! #5/23 15:11 @ mine -> 21:47 completed
        lc_d1_list  = [9, 4.5, 3, 2.25, 1.8, 1.5, 1.286, 1.125, 1]  # intra 05/24 08:34 completed
        lc_gpe_list = [9, 4.5, 3, 2.25, 1.8, 1.5, 1.286, 1.125, 1]  # intra 05/24 08:34 @ mine 12:48 completed
        
        # ↓ Change the parameter you want to modify to hoge_list[i-1], when changing lc_stn_list, intra d1=5, gpe=5 changes are mandatory
        self.stn_ratio    = 1 # Cerebral cortex - STN (D5 receptor)
        self.d1_ratio     = 1  # Cerebral cortex - D1 receptor
        self.d2_ratio     = 1 # Cerebral cortex - D2 receptor # Do not modify
        self.lc_stn_ratio = 1  # Local hyper-direct intra STN
        self.gl_stn_ratio = 1 # Global hyper-direct pathway inter STN 
        self.lc_gpe_ratio = 1 # Local indirect pathway intra GPE
        self.gl_gpe_ratio = 0.1 # Global indirect pathway inter GPE

        """BG_loop tools"""
        self.lc_hyper = np.zeros(self.shape) # Local hyper-direct pathway
        self.gl_hyper = np.zeros(self.shape) # Global hyper-direct pathway
        self.lc_gpe   = np.zeros(self.shape) # Local indirect pathway
        self.gl_gpe   = np.zeros(self.shape) # Global indirect pathway
        self.memory   = [] # Memory

        self.str  = np.zeros(self.shape)
        self.beta = 0.1


    def bg_loop_fh(self, state):
        """Retrieve location information"""
        i = state[0]
        j = state[1]
        s = tuple([i, j]) # Current cell
        u = tuple([i-1, j]) # Upper cell
        r = tuple([i, j+1]) # Right cell
        l = tuple([i, j-1]) # Left cell
        d = tuple([i+1, j]) # Lower cell
        
        """Retrieve goal positions"""
        goal1 = np.squeeze(np.where(self.maze=='G1'))
        goal2 = np.squeeze(np.where(self.maze=='G2'))
                                   
        

        """Reset cell activities"""
        self.x     = np.zeros(self.shape) # Cerebral cortex (input)
        self.stn   = np.zeros(self.shape) # Subthalamic nucleus (D5 receptor)
        self.d1    = np.zeros(self.shape) # D1 receptor
        self.d2    = np.zeros(self.shape) # D2 receptor
        self.gpe   = np.zeros(self.shape) # External segment of the globus pallidus
        self.snr   = np.zeros(self.shape + tuple([self.action_num + 1])) # Internal segment of the globus pallidus
        self.xs    = np.zeros(self.shape + tuple([self.action_num + 1])) # Cerebral cortex (re-input)
        self.strio = np.zeros(self.shape + tuple([self.action_num + 1])) # Striosome
        self.da    = np.zeros(self.shape) # Dopaminergic cells

        """Reset BG_loop tools"""
        self.lc_hyper = np.zeros(self.shape) # Local hyper-direct pathway
        self.gl_hyper = np.zeros(self.shape) # Global hyper-direct pathway
        self.lc_gpe   = np.zeros(self.shape) # Local indirect pathway (local GPE)
        self.gl_gpe   = np.zeros(self.shape) # Global indirect pathway (global GPE)

        self.str = np.zeros(self.shape)

        """Input"""
        self.x[s] = self.input
        self.x[u] = self.input
        self.x[r] = self.input
        self.x[l] = self.input
        self.x[d] = self.input

        if i == goal1[0] and j == goal1[1]:
            self.x[u] = 0
            self.x[r] = 0
            self.x[l] = 0
            self.x[d] = 0
            
        if i == goal2[0] and j == goal2[1]:
            self.x[u] = 0
            self.x[r] = 0
            self.x[l] = 0
            self.x[d] = 0

        """
        Hyper-direct pathway: Cerebral cortex → Subthalamic nucleus → Internal segment of the globus pallidus
        """
        self.stn      = self.x * self.wstn *self.stn_ratio
        self.lc_hyper = self.stn * self.lc_stn_ratio # Local hyper-direct pathway
        self.gl_hyper = self.stn * self.gl_stn_ratio # Global hyper-direct pathway
        self.snr[s+tuple([0])] = self.snr[s+tuple([0])] + self.lc_hyper[s] + self.gl_hyper[u]
        self.snr[s+tuple([1])] = self.snr[s+tuple([1])] + self.lc_hyper[s] + self.gl_hyper[r]
        self.snr[s+tuple([2])] = self.snr[s+tuple([2])] + self.lc_hyper[s] + self.gl_hyper[l]
        self.snr[s+tuple([3])] = self.snr[s+tuple([3])] + self.lc_hyper[s] + self.gl_hyper[d]
        self.snr[s+tuple([4])] = self.snr[s+tuple([4])] + self.lc_hyper[s]

        """
        Direct pathway: Cerebral cortex → D1 → [Inhibitory] Internal segment of the globus pallidus
        """
        self.d1 = self.x * self.wd1 * self.d1_ratio
        for p in range(0, 5):
            self.snr[s+tuple([p])] = self.snr[s+tuple([p])] - self.d1[s]

        """
        Indirect pathway: Cerebral cortex → D2 → [Inhibitory] External segment of the globus pallidus → [Inhibitory] Internal segment of the globus pallidus
        """
        self.d2  = self.x * self.wd2 * self.d2_ratio
        self.gpe = -self.d2 #* self.d2_ratio
        self.lc_gpe = self.gpe * self.lc_gpe_ratio # Local indirect pathway
        self.gl_gpe = self.gpe * self.gl_gpe_ratio # Global indirect pathway
        self.snr[s+tuple([0])] = self.snr[s+tuple([0])] - self.lc_gpe[s] - self.gl_gpe[u]
        self.snr[s+tuple([1])] = self.snr[s+tuple([1])] - self.lc_gpe[s] - self.gl_gpe[r]
        self.snr[s+tuple([2])] = self.snr[s+tuple([2])] - self.lc_gpe[s] - self.gl_gpe[l]
        self.snr[s+tuple([3])] = self.snr[s+tuple([3])] - self.lc_gpe[s] - self.gl_gpe[d]
        self.snr[s+tuple([4])] = self.snr[s+tuple([4])] - self.lc_gpe[s]

        """
        Internal segment of the globus pallidus → [Inhibitory] Cerebral cortex
        """
        if i==goal1[0] and j==goal1[1]: # Special processing when at the goal
            self.snr = self.snr - (self.lc_stn_ratio - self.d1_ratio + self.lc_gpe_ratio)
        elif i==goal2[0] and j==goal2[1]: # Special processing when at the goal
            self.snr = self.snr - (self.lc_stn_ratio - self.d1_ratio + self.lc_gpe_ratio)
        else:
            self.snr = self.snr - (self.lc_stn_ratio - self.d1_ratio + self.lc_gpe_ratio + self.gl_stn_ratio + self.gl_gpe_ratio)
            
        self.xs = - self.snr
        
    def policy(self, ep, state):
        """Retrieve location and action information"""
        s = tuple(state)
        u = np.array([-1, 0])
        r = np.array([0, 1])
        l = np.array([0, -1])
        d = np.array([1, 0])
        action = [u, r, l, d]

        """Policy based on ε-greedy"""
        self.epsilon = 0.5 * (1 / (ep + 1e-8))  
        if np.random.random() < self.epsilon:
            action = np.squeeze(random.choices(action))
            return action
        else:
            xx = np.array([self.snr[s+tuple([0])], self.snr[s+tuple([1])], self.snr[s+tuple([2])], self.snr[s+tuple([3])]])
            
            """softmax"""
            xx = xx * (1 + 0.001 * ep * int(ep/10)) * (1 + 0.01 * ep * int(ep/20)) * (1 + int(ep/280) * 10)
            
            c = np.max(xx)
            exp_x = np.exp(self.beta*(xx - c))
            sum_exp_x = np.sum(exp_x)
            p = exp_x / sum_exp_x
            max_index = np.random.choice(np.arange(len(xx)), p=p)
            
            return action[max_index]
    
    def bg_loop_sh(self, episode, a_cnt, action, state, reward):
        """Retrieve location and action information"""
        s = tuple([state[0], state[1]]) # Basal ganglia to be updated

        """
        Cerebral cortex → Striosome → [Inhibitory] Dopaminergic cells
        """
        self.strio[s] = self.xs[s] * self.wstrio[s]
        self.da[s]    = np.max(reward - self.strio[s])

        self.str[s] = np.max(-self.strio[s])

        """
        Synaptic plasticity via dopamine: Dopaminergic cells → Hyper-direct pathway, Direct pathway, Indirect pathway, Striosome
        """
        eta = 0.01
        self.wstn[s] += eta * self.da[s] * self.x[s]
        self.wd1[s] += eta * self.da[s] * self.x[s]
        self.wd2[s] += -eta * self.da[s] * self.x[s]
        
        self.memorize(episode, a_cnt, action, state, reward)


    def episode_fin(self, e):
        """Processing for each episode"""


    def memorize(self, ep, a_cnt, action, state, reward):
        directions = ["U", "D", "L", "R", "S"]
        sites = ["x", "D1", "D2", "lc_gpe", "gl_gpe", "lc_hyper", "gl_hyper", "SNr", "xs", "strio", "DA", "wstn"]
        
        buf = ({"episode":ep, "a_cnt":a_cnt, "action":action, "row":state[0], "col":state[1], "reward":reward, "DAsum":self.da.sum()})
        for site in sites:
            if site == "x":
                s = self.x
            elif site == "D1":
                s = self.d1
            elif site == "D2":
                s = self.d2
            elif site == "lc_gpe":
                s = self.lc_gpe
            elif site == "gl_gpe":
                s = self.gl_gpe
            elif site == "lc_hyper":
                s = self.lc_hyper
            elif site == "gl_hyper":
                s = self.gl_hyper
            elif site == "SNr":
                s = self.snr
            elif site == "xs":
                s = self.xs
            elif site == "strio":
                s = self.str
            elif site == "DA":
                s = self.da
            elif site == "wstn":
                s = self.wstn
                
            if site == "SNr" or site == "xs":
                for p in range(5):
                    for i in range(2, 9):
                        for j in range(2, 9):
                            buf.update({site + str(i) + str(j) + directions[p] : s[i, j, p]})
            else:
                for i in range(2, 9):
                    for j in range(2, 9):
                        buf.update({site + str(i) + str(j) : s[i, j]})
            self.memory.append(buf)
                                   
    def writer_csv(self,e, i):
        with open("sample{}-{}.csv".format(i, e), "w", newline="") as f:
            fieldnames = ['episode', 'a_cnt', 'action', 'row', 'col', 'reward', 'DAsum']+\
                ['x'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['D1'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['D2'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['lc_gpe'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['gl_gpe'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['lc_hyper'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['gl_hyper'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'U' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'R' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'L' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'D' for i in range(2,9) for j in range(2,9)]+\
                ['SNr'+str(i)+str(j)+'S' for i in range(2,9) for j in range(2,9)]+\
                ['xs'+str(i)+str(j)+'U' for i in range(2,9) for j in range(2,9)]+\
                ['xs'+str(i)+str(j)+'R' for i in range(2,9) for j in range(2,9)]+\
                ['xs'+str(i)+str(j)+'L' for i in range(2,9) for j in range(2,9)]+\
                ['xs'+str(i)+str(j)+'D' for i in range(2,9) for j in range(2,9)]+\
                ['xs'+str(i)+str(j)+'S' for i in range(2,9) for j in range(2,9)]+\
                ['strio'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['DA'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]+\
                ['wstn'+str(i)+str(j) for i in range(2,9) for j in range(2,9)]

            dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)
