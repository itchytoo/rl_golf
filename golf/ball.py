import numpy as np
import pygame
from ui import draw_ui


WHITE = (255, 255, 255)

class Ball:
    def __init__(self, x, y, radius, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.next_pos = None
        self.prev_pos = None
        self.start_pos = None
        self.animation_progress = 0
        self.animation_speed = 0.02
        self.locked_mouse_pos = None

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        if self.start_pos is not None and self.next_pos is not None:
            self._draw_dashed_line(screen, self.start_pos, (self.x, self.y))

    def move_to(self, x, y):
        self.prev_pos = (self.x, self.y)
        self.x = x
        self.y = y
        

    def get_pos(self):
        return np.array([self.x, self.y])

    def start_animation(self, next_pos, target_pos):
        self.start_pos = (self.x, self.y)
        self.next_pos = next_pos
        self.animation_progress = 0
        self.locked_mouse_pos = target_pos

    def update_animation(self):
        if self.next_pos is not None:
            self.animation_progress += self.animation_speed
            new_x = self.start_pos[0] + (self.next_pos[0] - self.start_pos[0]) * self.animation_progress
            new_y = self.start_pos[1] + (self.next_pos[1] - self.start_pos[1]) * self.animation_progress

            if self.animation_progress >= 1:
                self.animation_progress = 1
                self.move_to(*self.next_pos)    
                self.next_pos = None
                self.start_pos = None       

            self.move_to(new_x, new_y)

    def _draw_dashed_line(self, screen, start_pos, end_pos, dash_length=5):
        x1, y1 = start_pos
        x2, y2 = end_pos
        total_length = np.linalg.norm([x2 - x1, y2 - y1])
        num_dashes = int(total_length / dash_length)
        for i in range(num_dashes):
            start_fraction = i / num_dashes
            end_fraction = (i + 0.5) / num_dashes
            if end_fraction > 1:
                end_fraction = 1

            sx = x1 + (x2 - x1) * start_fraction
            sy = y1 + (y2 - y1) * start_fraction
            ex = x1 + (x2 - x1) * end_fraction
            ey = y1 + (y2 - y1) * end_fraction
            pygame.draw.line(screen, self.color, (sx, sy), (ex, ey), 2)

    def animate_path(self, screen, clock, course, aiming_system, button_rect, font, score, lie):
        while self.next_pos is not None:
            screen.fill(WHITE)
            course.draw(screen)
            self.draw(screen)
            draw_ui(screen, font, button_rect, aiming_system.current_club, score, lie)

            aiming_system.draw_arrow(screen, self.start_pos, self.locked_mouse_pos)
            aiming_system.draw_gaussian(screen, self.start_pos, self.locked_mouse_pos)

            self.update_animation()
            pygame.display.flip()
            clock.tick(120)

        pygame.time.wait(1000)
