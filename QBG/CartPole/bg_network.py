import numpy as np
import math
import random
import csv

"""BG Class"""

class BG_Network():

    def __init__(self, env):
        """Construct a network based on the environment"""
        self.action_num = 2 # Number of action choices
        self.shape      = (env, self.action_num) # Shape of the network

        """Cell settings"""
        self.x     = np.zeros(self.shape) # Cerebral cortex (input)
        self.stn   = np.zeros(self.shape) # Subthalamic nucleus (D5 receptor)
        self.d1    = np.zeros(self.shape) # D1 receptor
        self.d2    = np.zeros(self.shape) # D2 receptor
        self.gpe   = np.zeros(self.shape) # External segment of the globus pallidus
        self.snr_s = np.zeros(self.shape)
        self.snr_g = np.zeros(self.shape)
        self.xs_s  = np.zeros(self.shape)
        self.xs_g  = np.zeros(self.shape)
        self.strio = np.zeros(self.shape) # Striosome
        self.da    = np.zeros(self.shape) # Dopaminergic cells

        """Input"""
        self.input = 1 # Input

        """Weight parameters"""
        self.wstn   = np.ones(self.shape) # Cerebral cortex - Subthalamic nucleus (D5 receptor)
        self.wd1    = np.ones(self.shape) # Cerebral cortex - D1 receptor
        self.wd2    = np.ones(self.shape) # Cerebral cortex - D2 receptor
        self.wstrio = np.ones(self.shape)*5 # Cerebral cortex - Striosome
        self.w      = np.ones(self.shape + self.shape) # Cerebral cortex - Cerebral cortex

        """Size parameters"""
        self.stn_ratio    = 1 # Cerebral cortex - STN (D5 receptor)
        self.d1_ratio     = 1 # Cerebral cortex - D1 receptor
        self.d2_ratio     = 1 # Cerebral cortex - D2 receptor
        self.lc_stn_ratio = 1 # Local hyperdirect pathway
        self.gl_stn_ratio = 1 # Global hyperdirect pathway
        self.lc_gpe_ratio = 1 # Local indirect pathway
        self.gl_gpe_ratio = 0.1 # Global indirect pathway

        """BG_loop tools"""
        self.lc_hyper = np.zeros(self.shape) # Local hyperdirect pathway
        self.gl_hyper = np.zeros(self.shape) # Global hyperdirect pathway
        self.lc_gpe   = np.zeros(self.shape) # Local indirect pathway
        self.gl_gpe   = np.zeros(self.shape) # Global indirect pathway
        self.memory=[] # Memory


    def bg_loop_fh(self, state, pre_state, action, a_cnt):
        """Retrieve position information"""
        ps = tuple([pre_state])
        s  = tuple([state]) # Current position
        r  = tuple([state, 0]) # Right
        l  = tuple([state, 1]) # Left

        """Reset cell activity"""
        self.x     = np.zeros(self.shape) # Cerebral cortex (input)
        self.stn   = np.zeros(self.shape) # Subthalamic nucleus (D5 receptor)
        self.d1    = np.zeros(self.shape) # D1 receptor
        self.d2    = np.zeros(self.shape) # D2 receptor
        self.gpe   = np.zeros(self.shape) # External segment of the globus pallidus
        self.snr_s = np.zeros(self.shape)
        self.snr_g = np.zeros(self.shape)
        self.xs_s  = np.zeros(self.shape)
        self.xs_g  = np.zeros(self.shape)
        self.strio = np.zeros(self.shape) # Striosome
        self.da    = np.zeros(self.shape) # Dopaminergic cells

        """Reset BG_loop tools"""
        self.lc_hyper = np.zeros(self.shape) # Local hyperdirect pathway
        self.gl_hyper = np.zeros(self.shape) # Global hyperdirect pathway
        self.lc_gpe   = np.zeros(self.shape) # Local external segment of the globus pallidus (local indirect pathway)
        self.gl_gpe   = np.zeros(self.shape) # Global external segment of the globus pallidus (global indirect pathway)

        """Input"""
        self.x[s] = self.input

        """
        Hyperdirect pathway: Cerebral cortex → Subthalamic nucleus → Internal segment of the globus pallidus
        """
        self.stn      = self.x * self.wstn * self.stn_ratio
        self.lc_hyper = self.stn * self.lc_stn_ratio # Local hyperdirect pathway
        self.gl_hyper = self.stn * self.gl_stn_ratio # Global hyperdirect pathway

        self.snr_s[s] = self.snr_s[s] + self.lc_hyper[s]

        """
        Direct pathway: Cerebral cortex → D1 → [Inhibitory] Internal segment of the globus pallidus
        """
        self.d1 = self.x * self.wd1 * self.d1_ratio

        self.snr_s[s] = self.snr_s[s] - self.d1[s]

        """
        Indirect pathway: Cerebral cortex → D2 → [Inhibitory] External segment of the globus pallidus → [Inhibitory] Internal segment of the globus pallidus
        """
        self.d2 = self.x * self.wd2 * self.d2_ratio
        self.gpe = -self.d2
        self.lc_gpe = self.gpe * self.lc_gpe_ratio # Local indirect pathway
        self.gl_gpe = self.gpe * self.gl_gpe_ratio # Global indirect pathway

        self.snr_s[s] = self.snr_s[s] - self.lc_gpe[s]

        """
        Internal segment of the globus pallidus → [Inhibitory] Cerebral cortex
        """
        self.xs_s[s] = -np.round(self.snr_s[s] - (self.lc_stn_ratio), 10) # D1-D2 cancel each other out, so lc_stn_ratio is used for adjustment

    def policy(self, ep, state):
        """Retrieve position and action information"""
        r = tuple([state,0]) # Right
        l = tuple([state,1]) # Left

        """Policy based on ε-greedy"""
        self.epsilon = 0.5 * (1 / (ep + 1e-8)) 

        if np.random.random() < self.epsilon:
            action = np.squeeze(random.choices([0, 1]))
            return action
        else:
            xx = np.array([self.stn[r], self.stn[l]])
            max_x = [i for i, x in enumerate(xx) if x == max(xx)]
            action = np.squeeze(random.choices(max_x))
            return action


    def bg_loop_sh(self, pre_state, state, a_cnt, action, reward, ep, cart_pos, cart_v, pole_angle, pole_v, state_dim):
        """Retrieve position and action information"""
        ps = tuple([pre_state, action]) # Basal ganglia to update
        s = tuple([state])

        self.snr_g[ps] = self.snr_s[ps] + np.max((self.wstn[s]*self.stn_ratio*self.gl_stn_ratio)
            - (-self.wd2[s]*self.d2_ratio*self.gl_gpe_ratio)) # gl_stn - gl_gpe
        self.xs_g[ps] = -np.round(self.snr_g[ps] - (self.lc_stn_ratio+self.gl_gpe_ratio+self.gl_stn_ratio), 10)

        """
        Cerebral cortex → Striosome → [Inhibitory] Dopaminergic cells
        """
        self.strio[ps] = self.xs_g[ps] * self.wstrio[ps]
        self.da[ps] = reward - self.strio[ps]

        """
        Synaptic plasticity by dopamine: Dopaminergic cells → Hyperdirect pathway, Direct pathway, Indirect pathway
        """
        self.wstn[ps] += 0.01 * self.da[ps] * self.x[ps] 
        self.wd1[ps] += 0.01 * self.da[ps] * self.x[ps]
        self.wd2[ps] += -0.01 * self.da[ps] * self.x[ps]
        self.wstrio[ps] += 0.01 * self.da[ps] * self.x[ps]

        self.memorize(ep, a_cnt, pre_state, action, reward, self.wstn[ps], cart_pos, cart_v, pole_angle, pole_v, state_dim)


    def episode_fin(self, e):
        """Processing per episode"""


    def memorize(self, ep, a_cnt, pre_state, action, reward, q_value, cart_pos, cart_v, pole_angle, pole_v, state_dim):
        # Below, will be changed to loadtxt later due to length
        self.memory.append({"episode":ep, "a_cnt":a_cnt, "state":pre_state, "action":action, "reward":reward,\
                            "DAsum":self.da.sum(), "STN":q_value, "cart_pos":cart_pos,\
                            "cart_v":cart_v, "pole_angle":pole_angle, "pole_v":pole_v, "state_dim":state_dim
                           })


    def writer_csv(self,n):
        with open("sampleBG-{}.csv".format(n), "w", newline="") as f:
            fieldnames = ['episode','a_cnt','state','action','reward','DAsum','STN','cart_pos','cart_v','pole_angle','pole_v','state_dim']
            # ['snr'+str(i)+'R' for i in range(0,1296)]+\
            # ['snr'+str(i)+'L' for i in range(0,1296)]

            dict_writer = csv.DictWriter(f, fieldnames = fieldnames)
            dict_writer.writeheader()
            dict_writer.writerows(self.memory)
            
        np.savetxt("sampleBG_wstn-{}.csv".format(n), self.wstn, delimiter=",", header="l,r", comments="", fmt="%.6f")
