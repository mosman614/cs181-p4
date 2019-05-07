
# Imports.
import numpy as np
import numpy.random as npr
import pygame as pg
import math
import random
import matplotlib.pyplot as plt

from SwingyMonkey import SwingyMonkey


class Learner(object):
    '''
    This agent jumps randomly.
    '''

    def __init__(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.count = 0
        self.gravity = 0
        self.gamma = 0.2
        self.beta = 1
        self.alpha = [1]*864
        self.qtable = [0]*864
        self.currentindex = 0
        self.lastindex = 0
        self.best = 0
        self.tracker = [0]*864
        self.grav1 = []
        self.grav2 = []

    def reset(self):
        self.last_state  = None
        self.last_action = None
        self.last_reward = None
        self.count = 0
        self.currentindex = 0
        self.lastindex = 0
        self.best = 0

    def action_callback(self, state):
        '''
        Implement this function to learn things and take actions.
        Return 0 if you don't want to jump and 1 if you do.
        '''

        # You might do some learning here based on the current state and the last state.

        # You'll need to select and action and return it.
        # Return 0 to swing and 1 to jump.

        dct = state
        data_row = list(dct['tree'].values()) + list(dct['monkey'].values())
        print(data_row)
        q_input = []
        q_input.append(math.floor((data_row[0] +115) / 100.1))
        q_input.append(math.floor((data_row[4] - data_row[2] + 200)/104))
        if data_row[3] >= 20:
            q_input.append(5)
        if (data_row[3] >= 10) and (data_row[3] < 20):
            q_input.append(4)
        if (data_row[3] >= 0) and (data_row[3] < 10):
            q_input.append(3)
        if (data_row[3] >= -10) and (data_row[3] < 0):
            q_input.append(2)
        if (data_row[3] >= -20) and (data_row[3] < -10):
            q_input.append(1)
        if (data_row[3] < -20):
            q_input.append(0)
        if self.gravity == -4:
            q_input.append(1)
        if self.gravity != -4:
            q_input.append(0)

        self.currentindex = (q_input[0]*(1) + q_input[1]*(6) + q_input[2]*(36) + q_input[3]*(216))*2
        print("CURRENT INDEX")
        print(q_input)
        print(self.currentindex)

        self.tracker[self.currentindex]+=1
        self.alpha[self.currentindex] = (self.beta) / (self.beta + self.tracker[self.currentindex])

        if self.count != 0:
            if self.last_action == 0:
                self.qtable[self.lastindex] = (1 - self.alpha[self.lastindex])*self.qtable[self.lastindex] + (self.alpha[self.lastindex])*(self.last_reward + self.gamma*(max(self.qtable[self.currentindex], self.qtable[self.currentindex + 1])))
            if self.last_action == 1:
                self.qtable[self.lastindex + 1] = (1 - self.alpha[self.lastindex])*self.qtable[self.lastindex + 1] + (self.alpha[self.lastindex])*(self.last_reward + self.gamma*(max(self.qtable[self.currentindex], self.qtable[self.currentindex + 1])))

        self.lastindex = self.currentindex


        print(q_input)
        print("The index is")
        print(self.lastindex)
        print("Random Q-Table Entries")
        print(self.qtable[random.randint(1,863)])
        print(self.qtable[random.randint(1,863)])


        print("THE COUNT IS")
        print(self.count)

        if self.count == 1:
            self.gravity = dct['monkey']['vel']
            print("The gravity is")
            print(self.gravity) 
            self.count+=1


        if self.count == 0:
            self.last_state = state
            self.count+=1
            


        if self.qtable[self.currentindex] + 0.01 >= self.qtable[self.currentindex + 1]:
            self.best = 0
        if self.qtable[self.currentindex] + 0.01 < self.qtable[self.currentindex + 1]:
            self.best = 1
        print("Deciding between")
        print(self.qtable[self.currentindex], self.qtable[self.currentindex+ 1])


        #new_action = npr.rand() < 0.1
        new_state  = state

        self.last_action = self.best
        self.last_state  = new_state
        #print("Q-Table")
        #print(self.qtable)
        print(self.tracker)

        if data_row[5] > 300:
            return 0

        return self.last_action

    def reward_callback(self, reward):
        '''This gets called so you can see what reward you get.'''

        self.last_reward = reward
        print("REWARD IS")
        print(reward)



def run_games(learner, hist, iters = 100, t_len = 100):
    '''
    Driver function to simulate learning by having the agent play a sequence of games.
    '''
    for ii in range(iters):
        # Make a new monkey object.
        swing = SwingyMonkey(sound=False,                  # Don't play sounds.
                             text="Epoch %d" % (ii),       # Display the epoch on screen.
                             tick_length = t_len,          # Make game ticks super fast.
                             action_callback=learner.action_callback,
                             reward_callback=learner.reward_callback)

        # Loop until you hit something.
        while swing.game_loop():
            pass

        # Save score history.
        hist.append(swing.score)

        if swing.gravity == 1:
            learner.grav1.append(swing.score)
        else:
            learner.grav2.append(swing.score)
        # Reset the state of the learner.
        learner.reset()

    fig, ax = plt.subplots(1,2,figsize=(15,5))

    epochs_grav1 = list(range(1,len(learner.grav1)+1))
    epochs_grav2 = list(range(1,len(learner.grav2)+1))


    ax[0].plot(epochs_grav1, learner.grav1)
    ax[0].set_title("Gravity = 1")
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Score")

    ax[1].plot(epochs_grav2, learner.grav2)
    ax[1].set_title("Gravity = 4")
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Score")



    plt.show()
    # print('result: ')
    # print()
    # print(learner.grav1)
    # print()
    # print(learner.grav2)
    pg.quit()
    print(hist)
    #fig.show()
    return 


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 700, 2)

	# Save history. 
	np.save('hist',np.array(hist))
