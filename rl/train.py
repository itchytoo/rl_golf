from stable_baselines3.common.env_checker import check_env
import pygame
import sys
sys.path.append('/Users/guinnesschen/Desktop/234_final/gymnasium')
from golf_env import GolfGameEnv
from stable_baselines3 import PPO

pygame.init()
screen = pygame.display.set_mode((800, 500))

env = GolfGameEnv(player_profile="golf/profile.json", course_profile="golf/course.json", screen=screen)
check_env(env)

model = PPO("MultiInputPolicy", env, verbose=1, n_steps=8, batch_size=8)
model.learn(total_timesteps=80_000, progress_bar=True)

print("Saving model...")
model.save("ppo_golf")