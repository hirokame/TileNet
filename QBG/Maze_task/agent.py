from bg_network import BG_Network

class Agent():

    def __init__(self, maze_shape, action_num):
        self.bg_network = BG_Network(maze_shape, action_num)
 

    def learn_fh(self, state):
        self.bg_network.bg_loop_fh(state)


    def get_action(self, ep, state):
        action = self.bg_network.policy(ep, state)
        return action


    def learn_sh(self, ep, a_cnt, pre_state, state, action, reward):
        self.bg_network.bg_loop_sh(ep, a_cnt, pre_state, state, action, reward)


    def episode_fin(self, ep):
        self.bg_network.episode_fin(ep)


    def writer(self, epoch):
        self.bg_network.writer_csv(epoch)