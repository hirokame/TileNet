from agent import Agent
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
import seaborn as sns
import random

""" Environment class """
#url https://yottagin.com/?p=1557

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

    
    def set_out_wall(self):
        """ Create a 2D array that constitutes the entire maze, setting the outer perimeter as walls and the rest as paths. """
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
        """ Place poles inside the maze and knock them down in random directions.
        Place reference poles inside the outer perimeter at every other cell (coordinates where both x and y are even). """
        #random.seed(42) # Fix the maze with random numbers 
        for _x in range(2, self.width-1, 2):
            for _y in range(2, self.height-1, 2):
                self.maze[_x][_y] = self.WALL
                # Knock down the pole in a random direction to create a wall.
                # (However, avoid the following directions.)
                # Except for the inner walls of the first row, do not knock down the pole upwards.
                # If the pole has already been knocked down and turned into a wall, do not knock it down in that direction.
                while True:
                    if _y == 2:
                        direction = random.randrange(0, 4)
                    else:
                        direction = random.randrange(0, 3)
                    # Decide the direction to knock down the pole and create a wall.
                    wall_x = _x
                    wall_y = _y
                    # Right
                    if direction == 0:
                        wall_x += 1
                    # Down
                    elif direction == 1:
                        wall_y += 1
                    # Left
                    elif direction == 2:
                        wall_x -= 1
                    # Up
                    else:
                        wall_y -= 1
                    # If the direction to create a wall is not already a wall, create a wall.
                    if self.maze[wall_x][wall_y] != self.WALL:
                        self.maze[wall_x][wall_y] = self.WALL
                        break
        return self.maze

    ########################################################################
    def set_start_goal(self, start, goal):
        """ Place the start and goal in the maze. """
        self.maze[start[0]][start[1]] = 'S'
        self.maze[goal[0]][goal[1]] = 'G'
        for i in range(2,7):
            self.maze[3][i]=self.WALL
        self.maze[4][2]=self.WALL
        self.maze[5][2]=self.WALL
        self.maze[4][6]=self.WALL
        self.maze[5][6]=self.WALL
        return self.maze
    ########################################################################
    
    def bg_maze(self,kernel):
        """ Kernel adjustment for BG_network """
        if kernel>1:
            kernel-=1
            new_width=[['1']*self.width]*kernel
            new_width=np.array(new_width)
            new_width.reshape(self.width,kernel)

            new_height=[['1']*kernel]*(self.height+2*kernel)
            new_height=np.array(new_height)
            new_height.reshape(kernel,self.height+2*kernel)
            
            self.maze=np.vstack((new_width, self.maze))
            self.maze=np.vstack((self.maze, new_width)) 
            self.maze=np.hstack((self.maze,new_height))
            self.maze=np.hstack((new_height,self.maze))
        #self.maze is a str object
        self.maze=np.array(self.maze)
        print(self.maze)
        print()

    def reset(self):
        start = np.where(self.maze == 'S') # Get the index of the start
        start = list(map(list, start))
        self.start = np.array([start[0][0], start[1][0]]) 

    def run(self,agent,n,episode_count):
        """ Move the agent within the environment """ 
        self.reset() # Get and update the start position information
        a_list=[] # Log of action counts
        x_list=np.zeros([self.width, self.height]) # Log of position counts

        #print("Number of task trials:",n)
        for ep in range(1,episode_count+1):
            state = self.start
            pre_state = self.start
            done = False # Task completion judgment
            reward = 0
            a_cnt = 0 # Action count
            x_list[state[0]-2,state[1]-2]+=1 # Position count

            while True:
                action = agent.get_action(ep, state)
                a_cnt += 1 # Action count
                n_state, action, a_cnt, reward, done = self.step(agent, ep, state, action, a_cnt, done) # Advance the environment
                pre_state = state
                state = n_state
                agent.learn(ep,a_cnt,pre_state,state,action,reward)
                

                if done == True: # End judgment
                    agent.episode_fin(ep) # Update at the end of the episode
                    break

                x_list[state[0]-2,state[1]-2]+=1 # Position count
            a_list.append(a_cnt)

        """ Display results """
#         print(a_list) # Display the number of actions in each episode
#         x = np.arange(1, episode_count+1, 1) # Display the graph
#         y = a_list
#         plt.figure(figsize=(8, 4))
#         plt.plot(x, y, label='Label')
#         plt.legend()
#         plt.title('Change the number of actions')
#         plt.xlabel('Episode')
#         plt.ylabel('The number of actions')
#         plt.grid(True)
#         plt.show()

        agent.writer(n) # Write to CSV file
#         sns.heatmap(x_list, annot=False, square=True, cmap='Greens') # Display the heatmap of transition counts
#         plt.show()


    def step(self, agent, ep, state, action, a_cnt, done):
        """ Advance the environment """
        n_state = state + action
        if self.maze[n_state[0], n_state[1]] == "1": # If n_state is not reachable, choose an action again
            while self.maze[n_state[0], n_state[1]] == "1":
                action = agent.get_action(ep, state)
                n_state = state + action
#                 a_cnt += 1

        if self.maze[n_state[0], n_state[1]] == "G": # Determine the reward based on n_state
            reward=1
            done = True
        else:
            reward=0

        return n_state, action, a_cnt, reward, done