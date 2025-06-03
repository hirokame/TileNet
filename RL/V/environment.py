import numpy as np
import matplotlib.pyplot as plt
import random


""" Environment class """
#url https://yottagin.com/?p=1557

class Maze():
    PATH = 0
    WALL = 1

    def __init__(self, width, height):
        """Initial settings"""
        self.maze = []
        self.total_action_log = []
        self.width = width + 1
        self.height = height + 1
        self.pin = False

        if(self.height < 5 or self.width < 5): # The maze must be at least 5 in width and height to be generated
            print('at least 5')
            exit()
        if (self.width % 2) == 0:
            self.width += 1
        if (self.height % 2) == 0:
            self.height += 1

    
    def set_out_wall(self):
        """ Create a 2D array representing the maze, setting the outer perimeter as walls and the rest as paths. """
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
        Place reference poles inside the perimeter at every other cell (coordinates where both x and y are even). """
        #random.seed(42) # Fix the maze randomness
        for _x in range(2, self.width-1, 2):
            for _y in range(2, self.height-1, 2):
                self.maze[_x][_y] = self.WALL
                # Knock down poles in random directions to create walls.
                # (Except for the following cases.)
                # Poles inside the first row should not be knocked down upwards.
                # Poles should not be knocked down in directions where walls already exist.
                while True:
                    if _y == 2:
                        direction = random.randrange(0, 4)
                    else:
                        direction = random.randrange(0, 3)
                    # Decide the direction to knock down the pole to create a wall.
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
        
        #(7,7)
        for i in range(2,7):
            self.maze[3][i]=self.WALL
        self.maze[4][2]=self.WALL
        self.maze[5][2]=self.WALL
        self.maze[4][6]=self.WALL
        self.maze[5][6]=self.WALL
        
        #(1,6)
        for i in range(1,8):
            self.maze[2][i]=self.WALL
        self.maze[1][7]=self.WALL

        
        return self.maze
    ########################################################################
    
    def bg_maze(self,kernel):
        """ Adjust the kernel for BG_network """
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
        start = np.where(self.maze == 'S') # Get the index of the start position
        start = list(map(list, start))
        self.start = np.array([start[0][0], start[1][0]]) 

    def run(self,agent,n,episode_count):
        """ Move the agent within the environment """ 
        self.reset() # Get and update the start position information
        a_list=[] # Log of action counts
        x_list=np.zeros(self.maze.shape) # Log of position counts

        #print("Number of task trials:",n)
        for ep in range(1,episode_count+1):
            #print("episode:",e) 
            state=self.start 
            pre_state=state
            action=[0,1] # Dummy action
            a_cnt=0 # Action count
            reward=0 # Reward
            done=False # Task completion judgment
            i = 0
            while True:
                x_list[state[0],state[1]]+=1
                if done == True:
                    agent.episode_fin(ep) # Update at the end of the episode
                    break
                action=agent.get_action(ep,state) # Decide action with Policy
                a_cnt+=1 
                n_state,reward,done,action,a_cnt = self.step(agent,ep,state,action,done,a_cnt) # Advance the environment
                pre_state=state
                state=n_state
                agent.learn(ep,a_cnt,pre_state,state,action,reward,i) # BG_loop
                i += 1
#                 if i == 100:
#                    done = True
            a_list.append(a_cnt)
            agent.writer(n)

#         print(a_list)
        self.total_action_log.append(a_list)
        
        x = np.arange(1, episode_count+1, 1)
        y = a_list
        plt.figure(figsize=(8, 4))
        plt.plot(x, y, label='Label')
        plt.legend()
        plt.title('Change the number of actions')
        plt.xlabel('step')
        plt.ylabel('n_act')
        plt.grid(True)
        plt.show()

    def step(self, agent, ep, state, action, done, a_cnt):
        """ Advance the environment """
        n_state = state + action
        if self.maze[n_state[0], n_state[1]] == "1": # If n_state is not reachable, choose another action
            while self.maze[n_state[0], n_state[1]] == "1":
                action = agent.get_action(ep, state)
                n_state = state + action
                a_cnt+=1

        if self.maze[state[0], state[1]] == "G": # Decide reward based on n_state
            reward=1
            done = True
        else:
            reward=0

        return n_state, reward, done, action, a_cnt