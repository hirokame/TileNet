"""VBG"""
from bg_network2 import BG_Network        

class Agent():

    def __init__(self, env, i):
        self.bg_network2 = BG_Network(env, i)
 

    def learn_fh(self, state):
        self.bg_network2.bg_loop_fh(state)


    def get_action(self, ep, state):
        action = self.bg_network2.policy(ep, state)
        return action


    def learn_sh(self, ep, a_cnt, action, state, reward):
        self.bg_network2.bg_loop_sh(ep, a_cnt, action, state, reward)


    def episode_fin(self, ep):
        self.bg_network2.episode_fin(ep)


    def writer(self, e, i):
        self.bg_network2.writer_csv(e, i)