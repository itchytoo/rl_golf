import gymnasium as gym
from stable_baselines3 import PPO
import pygame
import sys
sys.path.append('/Users/guinnesschen/Desktop/234_final/gymnasium')
from golf_env import GolfGameEnv
sys.path.append('/Users/guinnesschen/Desktop/234_final/golf')
from constants import SCREEN_WIDTH, SCREEN_HEIGHT

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
env = GolfGameEnv(player_profile="golf/profile.json", course_profile="golf/course.json", screen=screen)
observation, _ = env.reset()
pygame.init()

print('loading model...')
model = PPO.load("rl/ppo_golf")

while True:
    action, _ = model.predict(observation)
    observation, reward, terminated, truncated, _ = env.step(action)
    env.render()
    if terminated:
        break

    if truncated:
        env.reset()

pygame.quit()


