from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import random

class Maze():
    PATH = 0
    WALL = 1

    def __init__(self, width, height):
        """Initial settings"""
        self.maze = []
        self.width = width + 1
        self.height = height + 1
        self.pin = False

        if(self.height < 5 or self.width < 5): # Maze must have width and height of at least 5 and be generated with odd numbers
            print("at least 5")
            exit()
        if (self.width % 2) == 0:
            self.width += 1
        if (self.height % 2) == 0:
            self.height += 1


    def set_out_wall(self):
        """Set the outer boundary of the maze as walls, and the rest as paths"""
        for _x in range(0, self.width):
            row = []
            for _y in range(0, self.height):
                if (_x == 0 or _y == 0 or _x == self.width-1 or _y == self.height -1):
                    cell = self.WALL
                else:
                    cell = self.PATH
                row.append(cell)
            self.maze.append(row)
            

    def set_start_goal(self, start, goal):
        """Place the start and goal in the maze"""
        self.maze[start[0]][start[1]] = "S"
        self.maze[goal[0]][goal[1]] = "G"
        
        # Set walls
        for i in range(2,7):
            self.maze[3][i]=self.WALL
        self.maze[4][2]=self.WALL
        self.maze[5][2]=self.WALL
        self.maze[4][6]=self.WALL
        self.maze[5][6]=self.WALL


    def bg_maze(self,kernel):
        """Adjust the kernel for the BG_network"""
        if kernel > 1:
            kernel -= 1
            new_width = [["1"] * self.width] * kernel
            new_width = np.array(new_width)
            new_width.reshape(self.width, kernel)

            new_height = [["1"] * kernel] * (self.height + 2 * kernel)
            new_height = np.array(new_height)
            new_height.reshape(kernel, self.height + 2 * kernel)
            
            self.maze = np.vstack((new_width, self.maze))
            self.maze = np.vstack((self.maze, new_width)) 
            self.maze = np.hstack((self.maze, new_height))
            self.maze = np.hstack((new_height, self.maze))

        self.maze = np.array(self.maze)
        print("maze_shape\n", self.maze)


    def reset(self):
        start = np.squeeze(np.where(self.maze == "S")) # Get the index of the start position
        row_ = self.maze.shape[0] - 4
        col_ = self.maze.shape[1] - 4
        return start, row_, col_


    def run(self, agent, epoch, episode_count):
        """Move the agent within the environment""" 
        start, row, col = self.reset() # Get and modify the start position information
        self.a_list = [] # Memory for action counts
        x_list = np.zeros([row,col]) # Memory for position counts
        

        for ep in tqdm(range(1, episode_count+1)):
            state = start # Initial position
            done = False # Task completion judgment
            a_cnt = -1 # Action count
            x_list[tuple(state-[2,2])] += 1 # Count position occurrences
            pre_state = state

            while True:
                agent.learn_fh(state) # bg_loop first half
                action = agent.get_action(ep,state) # Decide action using policy
                n_state, reward, done, action = self.step(agent, ep, state, action, done) # Advance the environment
                agent.learn_sh(ep, a_cnt, pre_state, state, action, reward) # bg_loop second half
                state = n_state

                x_list[tuple(state-[2,2])] += 1 # Count position occurrences
                a_cnt += 1 # Step count within one episode

                if done == True: # Completion judgment
                    agent.episode_fin(ep) # Update at the end of the episode
                    break
                    
#                 if a_cnt == 200:
#                     done = True

            self.a_list.append(a_cnt)
            
        
        """Count the shortest action steps in a_list"""
        min_a = np.count_nonzero(np.array(self.a_list) == 10)
        print(min_a)

        """Display results"""
        print(self.a_list) # Display action counts for each episode
        x = np.arange(1, episode_count+1, 1) # Display the graph
        y = self.a_list
        plt.figure(figsize=(8, 4))
        plt.plot(x, y)
        plt.title('learning_rate')
        plt.xlabel('Episode')
        plt.ylabel('action_steps')
        plt.grid(True)
        plt.show()

#         agent.writer(epoch) # Write to CSV file
        sns.heatmap(x_list, annot=False, square=True, cmap='Greens') # Display heatmap of transition counts
        plt.show()
        
        return a_cnt


    def step(self, agent, ep, state, action, done):
        """Advance the environment"""
        n_state = state + action
        if self.maze[n_state[0], n_state[1]] == "1": # If n_state is not reachable, choose another action
            while self.maze[n_state[0], n_state[1]] == "1":
                action = agent.get_action(ep, state)
                n_state = state + action

#         if self.maze[state[0], state[1]] == "G": # Determine reward based on n_state # Can also learn with state
#             reward=1
#             done = True
        if self.maze[n_state[0], n_state[1]] == "G": # Determine reward based on n_state
            reward=1
            done = True
        else:
            reward = 0 # reward = -0.01

        return n_state, reward, done, action