import gymnasium as gym
import pygame
from golf_env import GolfGameEnv
from constants import SCREEN_WIDTH, SCREEN_HEIGHT

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
env = GolfGameEnv(player_profile="golf/profile.json", course_profile="golf/course.json", screen=screen)
env.reset()
pygame.init()

while True:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, _ = env.step(action)

    env.render()
    if terminated:
        break

    if truncated:
        env.reset()

pygame.quit()