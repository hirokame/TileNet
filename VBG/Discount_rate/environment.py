from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from tqdm import tqdm

class Maze():
    PATH = 0
    WALL = 1

    def __init__(self, width, height):
        """Initial settings"""
        self.maze = []
        self.width = width + 1
        self.height = height + 1
        self.pin = False

        if(self.height < 5 or self.width < 5): # The maze must have a width and height of at least 5 and be generated with odd numbers
            print('at least 5')
            exit()
        if (self.width % 2) == 0:
            self.width += 1
        if (self.height % 2) == 0:
            self.height += 1
            
        self.A_list = []
        self.X_list = np.zeros([7,7])

    def set_out_wall(self):
        """Set the outer perimeter of the maze as walls, and everything else as paths"""
        for _x in range(0, self.width):
            row = []
            for _y in range(0, self.height):
                if (_x == 0 or _y == 0 or _x == self.width-1 or _y == self.height -1):
                    cell = self.WALL
                else:
                    cell = self.PATH
                row.append(cell)
            self.maze.append(row)
        return self.maze


    def set_inner_wall_boutaosi(self):
        """Generate walls
            # Place reference poles inside the outer perimeter at intervals of one cell, at even coordinates for both x and y
            # Knock down the poles in random directions to create walls
            # Except for the inner walls of the first row, poles must not be knocked down in the upward direction
            # If a pole has already been knocked down and turned into a wall, it must not be knocked down in that direction"""
        #random.seed(42) # Fix the maze with random numbers 
        for _x in range(2, self.width-1, 2):
            for _y in range(2, self.height-1, 2):
                self.maze[_x][_y] = self.WALL

                while True: # Decide the direction to knock down the pole and turn it into a wall
                    if _y == 2:
                        direction = random.randrange(0, 4)
                    else:
                        direction = random.randrange(0, 3)
                    wall_x = _x
                    wall_y = _y
                    if direction == 0:
                        wall_x += 1
                    elif direction == 1:
                        wall_y += 1
                    elif direction == 2:
                        wall_x -= 1
                    else:
                        wall_y -= 1

                    if self.maze[wall_x][wall_y] != self.WALL: # If the direction to turn into a wall is not already a wall, turn it into a wall
                        self.maze[wall_x][wall_y] = self.WALL
                        break
        return self.maze


    def set_start_goal(self, start, goal_1, goal_2):
        """Insert start and goal into the maze"""
        self.maze[start[0]][start[1]] = 'S'
        self.maze[goal_1[0]][goal_1[1]] = 'G1'
        self.maze[goal_2[0]][goal_2[1]] = 'G2'
        return self.maze


    def bg_maze(self,kernel):
        """Adjust kernel for BG_network"""
        if kernel > 1:
            kernel -= 1
            new_width = [['1'] * self.width] * kernel
            new_width = np.array(new_width)
            new_width.reshape(self.width, kernel)

            new_height = [['1'] * kernel] * (self.height + 2 * kernel)
            new_height = np.array(new_height)
            new_height.reshape(kernel, self.height + 2 * kernel)
            
            self.maze = np.vstack((new_width, self.maze))
            self.maze = np.vstack((self.maze, new_width)) 
            self.maze = np.hstack((self.maze, new_height))
            self.maze = np.hstack((new_height, self.maze))

        self.maze = np.array(self.maze)
        print(self.maze)
        print()


    def reset(self):
        start = np.where(self.maze == 'S') # Get the index of the start position
        start = list(map(list, start))
        self.start = np.array([start[0][0], start[1][0]]) 


    def run(self, agent, n, episode_count, i):
        """Move the agent within the environment""" 
        self.reset() # Get and update start position information
        self.get_reward1 = []
        self.get_reward2 = []
        a_list = [] # Memory for action count
        x_list = np.zeros([5, 5]) # Memory for position count
        xx_list = np.zeros([7,7])
        reward = 0

        for ep in range(1, episode_count+1):
#         for ep in tqdm(range(1, episode_count+1)):
            state = self.start
            done = False # Task completion judgment
            a_cnt = -1 # Action count
            x_list[state[0]-2, state[1]-2] += 1 # Position count
            pre_state = state
            
            if ep > episode_count-1:
                xx_list[state[0]-2, state[1]-2] += 1

            while True:
                agent.learn_fh(state) # bg_loop first half
                action = agent.get_action(ep,state) # Decide action with Policy
                a_cnt += 1 # Action count
                
                n_state, reward, done, action, a_cnt = self.step(agent, ep, state, action, done, a_cnt, episode_count, i) # Advance the environment                

                agent.learn_sh(ep, a_cnt, action, state, reward) # bg_loop second half
                state = n_state

                x_list[state[0]-2, state[1]-2] += 1 # Position count

                if done == True: # Completion judgment
                    agent.episode_fin(ep) # Update at the end of the episode
                    break
                                
                if ep > episode_count-1:
                    xx_list[state[0]-2, state[1]-2] += 1

            a_list.append(a_cnt)
            
        self.A_list.append(a_list)
        self.X_list += xx_list
        
#         print(n)
        
#         agent.writer(n, i) # Write to CSV file
#         self.result_plot(n, episode_count)
        
        return self.get_reward1, self.get_reward2 

        
    def result_plot(self, n, episode_count):
        """Display results"""
#         print(a_list) # Display the number of actions in each episode
#         print(np.mean(_list, axis=0))
        x = np.arange(1, episode_count+1, 1) # Display the graph
#         y = a_list
        y = np.mean(self.A_list, axis=0)
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Label')
        plt.legend()
        plt.title('Change the number of actions')
        plt.xlabel('Episode')
        plt.ylabel('The number of actions')
        plt.grid(True)
        plt.show()
        
        sns.heatmap(self.X_list, annot=False, square=True, cmap='Greens') # Display heatmap of transition counts
        plt.show()


    def step(self, agent, ep, state, action, done, a_cnt, episode_count, i):
        """Advance the environment"""
        n_state = state + action
        if self.maze[n_state[0], n_state[1]] == "1": # If n_state is not accessible, choose an action again
            while self.maze[n_state[0], n_state[1]] == "1":
                action = agent.get_action(ep, state)
                n_state = state + action

        if self.maze[state[0], state[1]] == "G1": # Decide reward based on n_state
            reward = 1
            done = True
            self.get_reward1.append(1)
            self.get_reward2.append(0)
        elif self.maze[state[0], state[1]] == "G2": # Decide reward based on n_state
            reward = 3
            done = True
            self.get_reward1.append(0)
            self.get_reward2.append(1)
        else:
            reward = 0

        return n_state, reward, done, action, a_cnt