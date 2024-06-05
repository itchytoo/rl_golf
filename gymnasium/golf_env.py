# ------------------------------------------------------------------------------------
# File: golf_env.py
# Authors: Guinness
# Date: 06/04/2024
# Description: This file contains the implementation of the GolfGameEnv class, which is a custom Gymnasium environment
# -------------------------------------------------------------------------------------

# import functions and classes
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import pygame
sys.path.append('/Users/guinnesschen/Desktop/234_final/golf')
from game import Game
from constants import SCREEN_WIDTH, SCREEN_HEIGHT

class GolfGameEnv(gym.Env):
    def __init__(self, player_profile, course_profile, screen):
        super().__init__()
        pygame.init()
        self.game = None
        # save player and course profiles
        self.player_profile = player_profile
        self.course_profile = course_profile
        self.screen = screen

        # define action and observation spaces and reward range
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        self. observation_space = spaces.Dict({
            "ball_position": spaces.Box(low=np.array([0, 0]), high=np.array([SCREEN_WIDTH, SCREEN_HEIGHT]), shape=(2,), dtype=np.int64),
            "lie": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int64),
            "course": spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8),
        })
        self.reward_range = (0, np.inf)
         
    def reset(self, seed=None):
        super().reset(seed=seed)
        #initialize game

        self.game = Game(self.screen, "golf/profile.json")
        while self.game.course.get_element_at(self.game.ball.get_pos()) != "Teebox":
            self.game.reset_game()

        # construct a one hot encoding of the lie
        lie = np.zeros(4)
        lie[0] = 1 # initial lie is always the teebox

        # construct the observation object
        observation = {
            "ball_position": self.game.ball.get_pos().astype(int),
            "lie": lie.astype(int), 
            "course": pygame.surfarray.array3d(self.game.course.course_surface)
        }

        return observation, {}

    def step(self, action):
        club_index, direction = action[0], action[1]
        # pre-process the action
        club_index = int((club_index + 1) * 6.5)
        direction = np.array([(direction + 1) * 180])

        club = self.game.clubs[club_index]
        # set up the aiming system with the selected club and lie
        self.game.aiming_system.change_club(club)
        current_lie = self.game.course.get_element_at(self.game.ball.get_pos().astype(int))
        self.game.aiming_system.set_lie(current_lie)

        # compute a target position based on the direction, where the magnitute is arbitrary
        target_pos = self.game.ball.get_pos() + np.array([np.cos(direction), np.sin(direction)]).squeeze(1) * 100

        # sample the next position of the ball
        next_pos = self.game.aiming_system.sample_gaussian(self.game.ball.get_pos(), target_pos)
        next_lie = self.game.course.get_element_at(next_pos.astype(int))

        # handle out of bounds and water hazards
        if next_lie == "Out of Bounds" or next_lie == "Water Hazard":
            # update game state
            self.game.score += 2
            # set the variables which will be returned by the step function
            reward = -2
            terminated = False
        # handle hole completion (end of the episode)
        elif next_lie == "Green":
            # update game state
            self.game.score += 1
            self.game.ball.move_to(*next_pos)
            # set the variables which will be returned by the step function
            reward = 10
            terminated = True
        else:
            # update game state
            self.game.score += 1
            self.game.ball.move_to(*next_pos)
            # set the variables which will be returned by the step function
            reward = -1
            terminated = False

        # construct a one hot encoding of the lie
        lie = np.zeros(4)
        lie[0] = 1 if next_lie == "Teebox" else 0
        lie[1] = 1 if next_lie == "Fairway" else 0
        lie[2] = 1 if next_lie == "Rough" else 0
        lie[3] = 1 if next_lie == "Bunker" else 0
        
        # construct the observation object
        observation = {
            "ball_position": self.game.ball.get_pos().astype(int),
            "lie": lie.astype(int),
            "course": pygame.surfarray.array3d(self.game.course.course_surface)
        }

        # truncate the episode if the score is too high
        if self.game.score > 20:
            truncated = True
        else:
            truncated = False

        return observation, reward, terminated, truncated, {}
        
    def render(self):
        clock = pygame.time.Clock()

        if self.game.ball.prev_pos is None:
            return
        # temporarily move the ball back to the previous position
        next_pos = self.game.ball.get_pos()
        self.game.ball.x, self.game.ball.y = self.game.ball.prev_pos[0], self.game.ball.prev_pos[1]

        # animate the ball moving to the current position
        self.game.ball.start_animation(next_pos, self.game.aiming_system.prev_target)
        self.game.ball.animate_path(self.game.screen, clock, self.game.course, self.game.aiming_system, self.game.button_rect, self.game.font, self.game.score, self.game.current_lie)

        # move the ball back to the current position
        self.game.ball.move_to(*next_pos)

    def close(self):
        pygame.quit()


    