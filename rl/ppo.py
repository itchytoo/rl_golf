import gymnasium as gym
import random
import torch
import numpy as np
import sys
import pygame
sys.path.append('/Users/guinnesschen/Desktop/234_final/gymnasium')
from golf_env import GolfGameEnv
sys.path.append('/Users/guinnesschen/Desktop/234_final/golf')
from constants import SCREEN_WIDTH, SCREEN_HEIGHT
sys.path.append('/Users/guinnesschen/Desktop/234_final/rl')
from ppo_agent import PPOAgent
from actor_cnn import ActorCnn, CriticCnn

#initialize pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

#make the environment
env = GolfGameEnv(player_profile="golf/profile.json", course_profile="golf/course.json", screen=screen)

#get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

#reset env
env.reset()

IMAGE_SHAPE = (3, 800, 500) # this is the size of our image... I added 3 here for RGB, but honestly IDK what input dim should be
BALL_POSITION_SHAPE = (2,) # this is the x, y position of the ball
LIE_SHAPE = (4,) # this is the lie of the ball
GAMMA = 0.99 # this is discount factor
ALPHA = 0.0001 # actor LR
BETA = 0.0001 # critic LR
TAU = 0.95 # for use in calculating PPO
BATCH_SIZE = 32
PPO_EPOCH = 5 # number of epochs for each call to model learn
CLIP_PARAM = 0.2 # hyperparam for PPO
UPDATE_EVERY = 1000 # how often we update network

# Create our agent
agent = PPOAgent(IMAGE_SHAPE, BALL_POSITION_SHAPE, LIE_SHAPE, device, GAMMA, ALPHA, BETA, TAU, UPDATE_EVERY, BATCH_SIZE, PPO_EPOCH, CLIP_PARAM, ActorCnn, CriticCnn)


def train(n_episodes=1000):
    for i_episode in range(1, n_episodes + 1): # Train for 1000 episodes
        state = env.reset() # Reset, aka start at beginning
        score = 0 #Set score to 0
        done = False
        num_steps = 0
        while True:
            print(f'num steps: {num_steps}')
            action, log_prob, value = agent.act(state) # Get predicted action and value, plus log_prob
            theta, club_index = action
            theta = np.array([theta])
            # create action dict
            action = {
                "club": club_index,
                "direction": theta
            }

            print(theta)

            next_state, reward, done = env.step(action) # Get next state, reward, and whether or not we are done from our Simulator
            score += reward # Add reward to score
            agent.step(state, action, value, log_prob, reward, done, next_state) # Take a step on the agent (may or may not update model)
            
            if done:
                break
            else:
                state = next_state

            num_steps += 1

        print(f"Episode {i_episode} achieved a score of {score}")

print('start training')
train()

# # TO VISUALIZE
# score = 0
# state = env.reset()
# done = False
# while True:
#     env.render()
#     action, _, _ = agent.act(state)
#     next_state, reward, done = env.step(action)
#     score += reward
#     state = next_state
#     if done:
#         print(f"Final score was {score}")
#         break
# env.close()
