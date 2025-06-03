import numpy as np
import random
import csv
import matplotlib.pyplot as plt
import seaborn as sns

class BG_Network():

    def __init__(self, maze_shape, action_num):
        """Construct a network based on the environment"""
        self.action_num = action_num # Number of action choices
        self.shape      = maze_shape.shape + tuple([self.action_num]) # Shape of the network
        

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
        self.stn_ratio    = 1 # Cerebral cortex - STN (D5 receptor)
        self.d1_ratio     = 1 # Cerebral cortex - D1 receptor
        self.d2_ratio     = 1 # Cerebral cortex - D2 receptor
        self.lc_stn_ratio = 1 # Local hyper-direct pathway
        self.gl_stn_ratio = 1 # Global hyper-direct pathway
        self.lc_gpe_ratio = 1 # Local indirect pathway
        self.gl_gpe_ratio = 0.1 # Global indirect pathway

        
        """BG_loop tools"""
        self.lc_hyper = np.zeros(self.shape) # Local hyper-direct pathway
        self.gl_hyper = np.zeros(self.shape) # Global hyper-direct pathway
        self.lc_gpe   = np.zeros(self.shape) # Local indirect pathway
        self.gl_gpe   = np.zeros(self.shape) # Global indirect pathway
        self.memory   = [] # Memory

        self.str = np.zeros(self.shape)


    def bg_loop_fh(self, state):
        """Retrieve position information"""
        i = state[0]
        j = state[1]
        s = tuple([i, j]) # Current cell
        u = tuple([i-1, j]) # Upper cell
        r = tuple([i, j+1]) # Right cell
        l = tuple([i, j-1]) # Left cell
        d = tuple([i+1, j]) # Lower cell

        """Reset cell activity"""
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
        self.lc_gpe   = np.zeros(self.shape) # Local external segment of the globus pallidus (local indirect pathway)
        self.gl_gpe   = np.zeros(self.shape) # Global external segment of the globus pallidus (global indirect pathway)

        self.str = np.zeros(self.shape)

        """Input"""
        self.x[s] = self.input
        self.x[u] = self.input
        self.x[r] = self.input
        self.x[l] = self.input
        self.x[d] = self.input

        """
        Hyper-direct pathway: Cerebral cortex → Subthalamic nucleus → Internal segment of the globus pallidus
        """
        self.stn      = self.x * self.wstn *self.stn_ratio
        self.lc_hyper = self.stn * self.lc_stn_ratio # Local hyper-direct pathway
        self.gl_hyper = self.stn * self.gl_stn_ratio # Global hyper-direct pathway
        for p in range(0, 4):
            self.snr[s+tuple([p, 4])] = self.snr[s+tuple([p, 4])] + self.lc_hyper[s+tuple([p])]
            self.snr[s+tuple([0, p])] = self.snr[s+tuple([0, p])] + self.lc_hyper[s+tuple([0])] + self.gl_hyper[u+tuple([p])]
            self.snr[s+tuple([1, p])] = self.snr[s+tuple([1, p])] + self.lc_hyper[s+tuple([1])] + self.gl_hyper[r+tuple([p])]
            self.snr[s+tuple([2, p])] = self.snr[s+tuple([2, p])] + self.lc_hyper[s+tuple([2])] + self.gl_hyper[l+tuple([p])]
            self.snr[s+tuple([3, p])] = self.snr[s+tuple([3, p])] + self.lc_hyper[s+tuple([3])] + self.gl_hyper[d+tuple([p])]

        """
        Direct pathway: Cerebral cortex → D1 → [Inhibitory] Internal segment of the globus pallidus
        """
        self.d1 = self.x * self.wd1 * self.d1_ratio
        for p in range(0, 4):
            for q in range(0, 5):
                self.snr[s+tuple([p, q])] = self.snr[s+tuple([p, q])] - self.d1[s+tuple([p])]

        """
        Indirect pathway: Cerebral cortex → D2 → [Inhibitory] External segment of the globus pallidus → [Inhibitory] Internal segment of the globus pallidus
        """
        self.d2     = self.x * self.wd2 * self.d2_ratio
        self.gpe    = -self.d2
        self.lc_gpe = self.gpe * self.lc_gpe_ratio # Local indirect pathway
        self.gl_gpe = self.gpe * self.gl_gpe_ratio # Global indirect pathway
        for p in range(0, 4):
            self.snr[s+tuple([p, 4])] = self.snr[s+tuple([p, 4])] - self.lc_gpe[s+tuple([p])]
            self.snr[s+tuple([0, p])] = self.snr[s+tuple([0, p])] - self.lc_gpe[s+tuple([0])] - self.gl_gpe[u+tuple([p])]
            self.snr[s+tuple([1, p])] = self.snr[s+tuple([1, p])] - self.lc_gpe[s+tuple([1])] - self.gl_gpe[r+tuple([p])]
            self.snr[s+tuple([2, p])] = self.snr[s+tuple([2, p])] - self.lc_gpe[s+tuple([2])] - self.gl_gpe[l+tuple([p])]
            self.snr[s+tuple([3, p])] = self.snr[s+tuple([3, p])] - self.lc_gpe[s+tuple([3])] - self.gl_gpe[d+tuple([p])]

        """
        Internal segment of the globus pallidus → [Inhibitory] Cerebral cortex
        """
        for p in range(0, 4):
            for q in range(0, 5):
                self.xs[s+tuple([p, q])] = -(self.snr[s+tuple([p, q])] - (self.lc_stn_ratio + self.gl_stn_ratio + self.gl_gpe_ratio))
                

    def policy(self, ep, state):
        """Retrieve position and action information"""
        s = tuple(state)
        u = np.array([-1, 0])
        r = np.array([0, 1])
        l = np.array([0, -1])
        d = np.array([1, 0])
        act_arr = [u, r, l, d]

        """Policy based on ε-greedy"""
        self.epsilon = 0.5 * (1 / (ep + 1e-8))
        if np.random.random() < self.epsilon: # ε-greedy
            action = np.squeeze(random.choices(act_arr))
            return action
        else: # greedy
            xx = np.array([np.max(self.stn[s+tuple([0])]), np.max(self.stn[s+tuple([1])]), np.max(self.stn[s+tuple([2])]), np.max(self.stn[s+tuple([3])])])

            max_x = [i for i, x in enumerate(xx) if x == max(xx)]
            max_x_index = np.squeeze(random.choices(max_x))
            return act_arr[max_x_index]


    def bg_loop_sh(self, ep, a_cnt, pre_state, state, action, reward):
        """Retrieve position and action information"""
        if action[0] == -1:
            a = 0
        elif action[1] == 1:
            a = 1
        elif action[1] == -1:
            a = 2
        elif action[0] == 1:
            a = 3
        s = tuple([state[0], state[1], a]) # Update basal ganglia

        """
        Cerebral cortex → Striosome → [Inhibitory] Dopaminergic cells
        """
        self.strio[s] = self.xs[s] * self.wstrio[s]
        self.da[s] = np.max(reward - self.strio[s])
        

        self.str[s] = np.max(-self.strio[s])

        """
        Synaptic plasticity by dopamine: Dopaminergic cells → Hyper-direct pathway, Direct pathway, Indirect pathway, Striosome
        """
        eta = 0.01 # Learning rate
        self.wstn[s] += eta * self.da[s] * self.x[s]
        self.wd1[s] += eta * self.da[s] * self.x[s]
        self.wd2[s] += -eta * self.da[s] * self.x[s]

        time=1
        self.memorize(ep,a_cnt,time,action,pre_state,reward)

    def episode_fin(self, e):
        """Processing per episode"""
        pass


    def memorize(self, ep, a_cnt, time, action, state, reward):
        
        directions = ['U', 'R', 'L', 'D', 'S']
        sites = ['x', 'D1', 'D2', 'lc_gpe', 'gl_gpe', 'lc_hyper', 'gl_hyper', 'SNr', 'xs', 'strio', 'DA']
        
        buf = ({'episode':ep, 'a_cnt':a_cnt, 'time':time, 'action':action, 's_row':state[0], 's_col':state[1],'reward':reward, 'DAsum':self.da.sum()})
        for site in sites:
            if site == 'x':
                s = self.x
            elif site == 'D1':
                s = self.d1
            elif site == 'D2':
                s = self.d2
            elif site == 'lc_gpe':
                s = self.lc_gpe
            elif site == 'gl_gpe':
                s = self.gl_gpe
            elif site == 'lc_hyper':
                s = self.lc_hyper
            elif site == 'gl_hyper':
                s = self.gl_hyper
            elif site == 'SNr':
                s = self.snr
            elif site == 'xs':
                s = self.xs
            elif site == 'strio':
                s = self.str
            elif site == 'DA':
                s = self.da
            
            if site == 'SNr' or site == 'xs':
                for q in range(0, 5): # directions(URLDS)
                    for i in range(2, 9):
                        for j in range(2, 9):
                            for p in range(0, 4):# directions(URLD)
                                buf.update({site + str(i) + str(j) + directions[p] + directions[q] : s[i, j, p, q]})
            else:
                for i in range(2, 9):
                    for j in range(2, 9):
                        for p in range(0, 4):
                            buf.update({site + str(i) + str(j) + directions[p] : s[i, j, p]})
        
        self.memory.append(buf)
        
    def named_arr(self, name, row, col):
        directions_1 = ["U", "R", "L", "D"]
        directions_2 = ["U", "R", "L", "D", "S"]
        tmp_arr = []

        if name == "SNr" or name == "xs":
            for direction_1 in directions_1:
                for direction_2 in directions_2:
                    tmp_arr += [name+str(i)+str(j)+direction_1+direction_2 for i in range(row, col) for j in range(row, col)]
            return tmp_arr

        else:    
            for direction in directions_1:
                tmp_arr += [name+str(i)+str(j)+direction for i in range(row, col) for j in range(row, col)] 
            return tmp_arr
    
    def writer_csv(self, epoch):
        row = 2
        col = 9
        names = ["x", "D1", "D2", "lc_gpe", "gl_gpe", "lc_hyper", "gl_hyper", "SNr", "xs", "strio", "DA"]
        for name in names:
            if name == "x":
                x_ar = self.named_arr(name, row, col)
            elif name == "D1":
                d1_ar = self.named_arr(name, row, col)
            elif name == "D2":
                d2_ar = self.named_arr(name, row, col)
            elif name == "lc_gpe":
                lc_gpe_ar = self.named_arr(name, row, col)
            elif name == "gl_gpe":
                gl_gpe_ar = self.named_arr(name, row, col)
            elif name == "lc_hyper":
                lc_hyper_ar = self.named_arr(name, row, col)
            elif name == "gl_hyper":
                gl_hyper_ar = self.named_arr(name, row, col)
            elif name == "SNr":
                SNr_ar = self.named_arr(name, row, col)
            elif name == "xs":
                xs_ar = self.named_arr(name, row, col)
            elif name == "strio":
                strio_ar = self.named_arr(name, row, col)
            elif name == "DA":
                da_ar = self.named_arr(name, row, col)
        
        with open("sampleA_200step-{}.csv".format((epoch,self.gl_stn_ratio,self.gl_gpe_ratio)), "w", newline="") as f:
            fieldnames = ['episode', 'a_cnt', 'time', 'action', 's_row', 's_col', 'reward', 'DAsum']+\
                x_ar+d1_ar+d2_ar+lc_gpe_ar+gl_gpe_ar+lc_hyper_ar+gl_hyper_ar+SNr_ar+xs_ar+strio_ar+da_ar
                
            dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)
