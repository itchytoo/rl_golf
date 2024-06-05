import pygame
from course import GolfCourse
from ball import Ball
import json
from ui import draw_ui, draw_out_of_bounds, draw_hole_complete
from aiming import AimingSystem
from constants import *


class Game:
    def __init__(self, screen, profile_file):
        # set UI params
        self.font = pygame.font.Font(None, 36)
        self.button_rect = pygame.Rect((SCREEN_WIDTH - BUTTON_WIDTH) // 2, SCREEN_HEIGHT - BUTTON_HEIGHT - 20, BUTTON_WIDTH, BUTTON_HEIGHT)
        # set game params   
        self.screen = screen
        self.load_profile(profile_file)
        # initialize game objects
        self.reset_game()

    def reset_game(self):
        # generate a new random course
        self.course = GolfCourse(par=4, difficulty=2)
        # place the ball at the teebox
        start_pos = self.course.teebox.rotated_rect.center
        self.ball = Ball(start_pos[0], start_pos[1], 3, WHITE)
        # initialize the aiming system according to the player's profile
        self.aiming_system = AimingSystem(params=self.profile)
        # reset game state
        self.score = 0
        self.done = False
        self.current_club_index = 0
        self.current_lie = 'Teebox'

    def load_profile(self, profile_file):
        with open(profile_file, 'r') as file:
            self.profile = json.load(file)
        self.clubs = list(self.profile.keys())

    def draw(self, mouse_pos):
        # draw the primary game elements
        self.course.draw(self.screen)
        self.ball.draw(self.screen)
        draw_ui(self.screen, self.font, self.button_rect, self.aiming_system.current_club, self.score, self.current_lie)

        if self.done:
            # draw the hole complete message
            draw_hole_complete(self.screen, self.font)

        else:
            # draw the aiming system
            if self.ball.next_pos is None:
                self.aiming_system.draw_arrow(self.screen, self.ball.get_pos(), mouse_pos)
                self.aiming_system.draw_gaussian(self.screen, self.ball.get_pos(), mouse_pos)
            else:
                self.aiming_system.draw_arrow(self.screen, self.ball.get_pos(), self.ball.locked_mouse_pos)
                self.aiming_system.draw_gaussian(self.screen, self.ball.get_pos(), self.ball.locked_mouse_pos)

    def handle_event(self, event, mouse_pos):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # check if the reset game button was clicked
            if self.button_rect.collidepoint(event.pos):
                self.reset_game()

            # otherwise, treat the click as a shot
            else:
                # if the game is done, don't allow the player to take any more shots
                if self.done:
                    return

                # sample a target position based on the aiming system
                next_pos = self.aiming_system.sample_gaussian(self.ball.get_pos(), mouse_pos)

                # if the ball went out of bounds or into water, handle it
                next_lie = self.course.get_element_at(next_pos.astype(int))
                if next_lie == 'Out of Bounds' or next_lie == 'Water Hazard':
                    self.handle_out_of_bounds(next_pos, next_lie)

                # if the ball ended up on the green, then the hole is complete
                elif next_lie == 'Green':
                    # animate the ball travelling to the green
                    self.ball.start_animation(next_pos, mouse_pos)
                    self.ball.animate_path(self.screen, pygame.time.Clock(), self.course, self.aiming_system, self.button_rect, self.font, self.score, self.current_lie)
                    # more hole completion logic
                    self.score += 1
                    self.current_lie = self.course.get_element_at(self.ball.get_pos().astype(int))
                    self.done = True
                
                # otherwise, we just move the ball to the target position
                else:
                    # animate the ball travelling to the target position
                    self.ball.start_animation(next_pos, mouse_pos)
                    self.ball.animate_path(self.screen, pygame.time.Clock(), self.course, self.aiming_system, self.button_rect, self.font, self.score, self.current_lie)
                    # update the current lie and score
                    self.score += 1
                    self.current_lie = self.course.get_element_at(self.ball.get_pos().astype(int))
                    self.aiming_system.set_lie(self.current_lie)
        
        # listen for key presses to change clubs
        elif event.type == pygame.KEYDOWN:
            # if the game is done, don't allow the player to switch clubs
            if self.done:
                return

            # otherwise, cycle through the clubs based on the left and right arrow keys
            if event.key == pygame.K_LEFT:
                self.current_club_index = (self.current_club_index - 1) % len(self.clubs)
                self.aiming_system.change_club(self.clubs[self.current_club_index])
            elif event.key == pygame.K_RIGHT:
                self.current_club_index = (self.current_club_index + 1) % len(self.clubs)
                self.aiming_system.change_club(self.clubs[self.current_club_index])

    def handle_out_of_bounds(self, target_pos, next_lie):
        # save the previous position
        previous_pos = self.ball.get_pos() 
        # animate the ball travelling to the target position
        self.ball.start_animation(target_pos, self.aiming_system.prev_target)
        self.ball.animate_path(self.screen, pygame.time.Clock(), self.course, self.aiming_system, self.button_rect, self.font, self.score, self.current_lie)
        # increment the score and display the out of bounds message
        self.score += 2
        self.current_lie = next_lie
        draw_out_of_bounds(self.screen, self.font)
        pygame.display.flip()
        # wait for a second before moving the ball back
        pygame.time.wait(1000)
        # go back to the previous position
        self.ball.move_to(*previous_pos)
        self.current_lie = self.course.get_element_at(self.ball.get_pos().astype(int))
        
