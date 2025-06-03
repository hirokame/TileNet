from bg_network import BG_Network

"""Agent Class for CartPole Environment"""

class Agent():
    def __init__(self,env):
        self.bg_network = BG_Network(env)


    def learn_fh(self, state, pre_state, action, a_cnt):
        self.bg_network.bg_loop_fh(state, pre_state, action, a_cnt)


    def get_action(self, ep, state):
        action = self.bg_network.policy(ep, state)
        return action


    def learn_sh(self, pre_state, state, a_cnt, action, reward, episode, cart_pos, cart_v, pole_angle, pole_v, state_dim):
        self.bg_network.bg_loop_sh(pre_state, state, a_cnt, action, reward, episode, cart_pos, cart_v, pole_angle, pole_v, state_dim)


    def episode_fin(self, ep):
        self.bg_network.episode_fin(ep)


    def writer(self, n):
        self.bg_network.writer_csv(n)