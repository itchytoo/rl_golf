# ------------------------------------------------------------------------------------
# File: course.py
# Authors: Guinness
# Date: 06/04/2023
# Description: This file contains the CourseElement, Teebox, and GolfCourse classes. The CourseElement class is an abstract class that represents a course element. 
# The Teebox class is a subclass of CourseElement that represents a teebox. The GolfCourse class represents a golf course and contains methods 
# -------------------------------------------------------------------------------------

# import packagees
import pygame
import numpy as np
from utils import generate_bezier_path, generate_height_envelope
import random
from constants import *

# ---------------
# Class Definitions
# ---------------

class CourseElement:
    def __init__(self, rect, color):
        self.rect = rect
        self.color = color

    def draw(self, screen):
        # override this method in subclasses
        pass

class Teebox(CourseElement):
    def __init__(self, rect, angle):
        super().__init__(rect=rect, color=GREEN_TEEBOX)
        self.angle = angle

        # precompute the rotated rectangle for the teebox
        teebox_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        teebox_surface.fill(self.color)
        self.rotated_teebox = pygame.transform.rotate(teebox_surface, -self.angle)
        self.rotated_rect = self.rotated_teebox.get_rect(center=self.rect.center)

    def draw(self, screen):
        screen.blit(self.rotated_teebox, self.rotated_rect.topleft)


class Green(CourseElement):
    def __init__(self, rect, angle):
        super().__init__(rect, GREEN_GREEN)
        self.angle = angle

        # randomly generate the position of the hole. The hole must be within the rotated ellipse bounded by the rotated rectangle.
        # this calculation is slighlty involved, but we we'll do it via rejection sampling via the mathematical definition of an ellipse
 
        while True:
            hole_x = random.randint(rect.left + HOLE_MARGIN, rect.right - HOLE_MARGIN)
            hole_y = random.randint(rect.top + HOLE_MARGIN, rect.bottom - HOLE_MARGIN)

            # check if these coordinates are within the ROTATED ellipse
            x = hole_x - rect.centerx
            y = hole_y - rect.centery
            a = rect.width / 2
            b = rect.height / 2
            angle = np.radians(self.angle)
            ellipse = (x * np.cos(-angle) + y * np.sin(-angle))**2 / a**2 + (x * np.sin(-angle) - y * np.cos(-angle))**2 / b**2
            if ellipse <= 0.85: # 0.95 is a fudge factor to ensure the hole is not too close to the edge of the green
                break

        self.hole_position = (hole_x, hole_y)

    def draw(self, screen):
        # draw a rotated ellipse bounded by the rectangle
        green_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(green_surface, self.color, green_surface.get_rect())
        
        # rotate the ellipse
        rotated_green = pygame.transform.rotate(green_surface, -self.angle)
        screen.blit(rotated_green, self.rect.topleft)

        # draw the hole
        self.draw_hole(screen)

    def draw_hole(self, screen):
        hole_x, hole_y = self.hole_position
        pygame.draw.circle(screen, BLACK, (hole_x, hole_y), HOLE_RADIUS)
        flag_pole_top = (hole_x, hole_y - FLAG_HEIGHT)
        pygame.draw.line(screen, BLACK, (hole_x, hole_y), flag_pole_top, 2)
        flag_points = [(hole_x, hole_y - FLAG_HEIGHT), (hole_x + FLAG_WIDTH, hole_y - FLAG_HEIGHT + FLAG_WIDTH // 2), (hole_x, hole_y - FLAG_HEIGHT + FLAG_WIDTH)]
        pygame.draw.polygon(screen, RED, flag_points)

class Hazard(CourseElement):
    def __init__(self, rect, color, angle):
        super().__init__(rect, color)
        self.angle = angle

    def draw(self, screen):
        # draw a rotated rectangle for the hazard
        hazard_surface = pygame.Surface((self.rect.width, self.rect.height), pygame.SRCALPHA)
        pygame.draw.ellipse(hazard_surface, self.color, hazard_surface.get_rect())
        rotated_hazard = pygame.transform.rotate(hazard_surface, -self.angle)

        rotated_rect = rotated_hazard.get_rect(center=self.rect.center)
        screen.blit(rotated_hazard, rotated_rect.topleft)


class FairwayAndRough(CourseElement):
    def __init__(self, rect):
        super().__init__(rect=rect, color=GREEN_FAIRWAY)

        # generate random points for the bezier curve
        points = self.generate_path(rect)
        self.fairway_path = self.generate_fairway_path(points)
        self.rough_path = self.generate_rough_path(points)
        
        self.fairway_envelope = self.generate_fairway_bounds()
        self.rough_envelope = self.generate_rough_bounds()

    def generate_path(self, rect, num_points=4):
        # randomly sample the points for the bezier curve
        points = []
        for i in range(num_points):
            x = rect.midleft[0] + (i) * (rect.width / (num_points - 1))
            y = random.randint(rect.top + COURSE_VERTICAL_MARGIN, rect.bottom - COURSE_VERTICAL_MARGIN)
            points.append([x, y])
        return np.array(points)

    def generate_fairway_path(self, points):
        # generate the path using the bezier curve helper function
        path = generate_bezier_path(points)
        return path

    def generate_rough_path(self, points):
        # Shift the first and last point by the rough margin
        points[0][0] -= ROUGH_MARGIN
        points[-1][0] += ROUGH_MARGIN
        # generate the path using the bezier curve helper function
        path = generate_bezier_path(points)
        return path

    def generate_fairway_bounds(self):
        # generate the bounds of the fairway for each point in the path
        envelope_deltas = []
        for point in self.fairway_path:
            delta_y = generate_height_envelope(point, self.fairway_path[0], self.fairway_path[-1])
            envelope_deltas.append(delta_y)
        return envelope_deltas           
         
    def generate_rough_bounds(self):
        # generate the bounds of the rough for each point in the path
        envelope_deltas = []
        for point in self.rough_path:
            delta_y = generate_height_envelope(point, self.rough_path[0], self.rough_path[-1])
            envelope_deltas.append(delta_y)
        return envelope_deltas  

    def draw_fairway(self, screen):
        # create a surface for the fairway 
        fairway_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_WIDTH), pygame.SRCALPHA)

        # draw overlapping rectangles for the fairway and rough
        for i in range(len(self.fairway_path) - 1):
            p0 = self.fairway_path[i] 
            p1 = self.fairway_path[i + 1] 
            angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            delta_y = self.fairway_envelope[i]

            # calculate the width of the rectangle
            width = 10
            fairway_height = 2 * delta_y

            # create a temporary surface for the rectangle
            temp_surface_fairway = pygame.Surface((width, fairway_height), pygame.SRCALPHA)
            temp_surface_fairway.fill(GREEN_FAIRWAY)

            # rotate the temporary surface
            rotated_surface_fairway = pygame.transform.rotate(temp_surface_fairway, -angle)

            # get the new rect of the rotated surface
            rotated_rect_fairway = rotated_surface_fairway.get_rect(center=(p0[0], p0[1]))

            # blit the rotated surface onto the fairway surface
            fairway_surface.blit(rotated_surface_fairway, rotated_rect_fairway.topleft)
      
        # blit the surfaces to the screen
        screen.blit(fairway_surface, (0,0))

    def draw_rough(self, screen):
        # create a surface for the rough
        rough_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_WIDTH), pygame.SRCALPHA)

        # draw overlapping rectangles for the fairway and rough
        for i in range(len(self.rough_path) - 1):
            p0 = self.rough_path[i] 
            p1 = self.rough_path[i + 1] 
            angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
            delta_y = self.rough_envelope[i]

            # calculate the width of the rectangle
            width = 10
            rough_height = 2 * delta_y * ROUGH_HEIGHT_MULTIPLIER

            # create a temporary surface for the rectangle
            temp_surface_rough = pygame.Surface((width, rough_height), pygame.SRCALPHA)
            temp_surface_rough.fill(GREEN_ROUGH)

            # rotate the temporary surface
            rotated_surface_rough = pygame.transform.rotate(temp_surface_rough, -angle)

            # get the new rect of the rotated surface
            rotated_rect_rough = rotated_surface_rough.get_rect(center=(p0[0], p0[1]))

            # blit the rotated surface onto the fairway surface
            rough_surface.blit(rotated_surface_rough, rotated_rect_rough.topleft)
      
        # blit the surfaces to the screen
        screen.blit(rough_surface, (0,0))

    def draw(self, screen):
        self.draw_rough(screen)
        self.draw_fairway(screen)

class GolfCourse:
    def __init__(self, par, difficulty):
        self.par = par
        self.difficulty = difficulty
        self.fairway_and_rough = FairwayAndRough(pygame.Rect(100, 100, SCREEN_WIDTH-200, SCREEN_HEIGHT-200))
        
        self.initialize_teebox()
        self.initialize_green()
        self.initialize_hazards()

        # draw the elements to the course surface
        self.course_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.fairway_and_rough.draw(self.course_surface)
        self.teebox.draw(self.course_surface)
        self.green.draw(self.course_surface)
        
        # draw the hazards
        for bunker in self.bunkers:
            bunker.draw(self.course_surface)
        for water_hazard in self.water_hazards:
            water_hazard.draw(self.course_surface)

    def initialize_teebox(self):
        # calculate the position and angle of the teebox
        teebox_angle = np.degrees(np.arctan2(self.fairway_and_rough.fairway_path[1][1] - self.fairway_and_rough.fairway_path[0][1], self.fairway_and_rough.fairway_path[1][0] - self.fairway_and_rough.fairway_path[0][0]))
        teebox_x = self.fairway_and_rough.fairway_path[0][0] + TEEBOX_MARGIN * np.cos(np.radians(teebox_angle))
        teebox_y = self.fairway_and_rough.fairway_path[0][1] + TEEBOX_MARGIN * np.sin(np.radians(teebox_angle))

        # create the teebox
        self.teebox = Teebox(pygame.Rect(teebox_x, teebox_y - TEEBOX_HEIGHT / 2, TEEBOX_WIDTH, TEEBOX_HEIGHT), teebox_angle)

    def initialize_green(self):
        # calculate the position of the green
        green_angle = np.degrees(np.arctan2(self.fairway_and_rough.fairway_path[-1][1] - self.fairway_and_rough.fairway_path[-2][1], self.fairway_and_rough.fairway_path[-1][0] - self.fairway_and_rough.fairway_path[-2][0]))
        green_x = self.fairway_and_rough.fairway_path[-1][0] - GREEN_MARGIN * np.cos(np.radians(green_angle))
        green_y = self.fairway_and_rough.fairway_path[-1][1] - GREEN_MARGIN * np.sin(np.radians(green_angle))

        # generate random dimensions for the green
        green_width = GREEN_WIDTH + random.randint(-10, 50)
        green_height = GREEN_HEIGHT + random.randint(-10, 50)

        # create the green
        self.green = Green(pygame.Rect(green_x - green_width / 2, green_y - green_height / 2, green_width, green_height), green_angle)

    def initialize_hazards(self):
        # sample the number of each type of hazard
        num_bunkers = max(0, self.difficulty * self.par // 2 - 1)
        num_water_hazards = self.difficulty * self.par // 6

        # create the hazards
        self.bunkers = []
        self.water_hazards = []

        # helper function to get a random point along the fairway path
        def get_random_point_along_fairway(max_offset=55, min_offset=20):
            start_index = len(self.fairway_and_rough.fairway_path) // 4
            end_index = int(2.75 * len(self.fairway_and_rough.fairway_path) // 4)
            index = random.randint(start_index, end_index)
            point = self.fairway_and_rough.fairway_path[index]
            offset = random.uniform(min_offset, max_offset)
            offset *= random.choice([-1, 1])
            return [point[0] + offset, point[1] + offset]

        # helper function to check if a hazard is in a valid position
        def is_valid_hazard_position(rect):
            for bunker in self.bunkers:
                if bunker.rect.colliderect(rect):
                    return False
            for water_hazard in self.water_hazards:
                if water_hazard.rect.colliderect(rect):
                    return False
            return True

        for _ in range(num_bunkers):
            while True:
                bunker_width = random.randint(20, 90)
                bunker_height = random.randint(20, 55)
                bunker_position = get_random_point_along_fairway()
                bunker_angle = random.randint(0, 360)

                bunker_rect = pygame.Rect(bunker_position[0] - bunker_width / 2, bunker_position[1] - bunker_height / 2, bunker_width, bunker_height)
                if is_valid_hazard_position(bunker_rect):
                    self.bunkers.append(Hazard(bunker_rect, YELLOW, bunker_angle))
                    break

        for _ in range(num_water_hazards):
            while True:
                water_width = random.randint(60, 120)
                water_height = random.randint(50, 100)
                water_position = get_random_point_along_fairway()
                water_angle = random.randint(0, 360)

                water_rect = pygame.Rect(water_position[0] - water_width / 2, water_position[1] - water_height / 2, water_width, water_height)
                if is_valid_hazard_position(water_rect):
                    self.water_hazards.append(Hazard(water_rect, BLUE, water_angle))
                    break
    
    def draw(self, screen):
        # draw the course surface to the main screen
        screen.blit(self.course_surface, (0, 0))

    def get_element_at(self, pos):
        # check if pos is out of the screen
        if pos[0] < 0 or pos[0] >= SCREEN_WIDTH or pos[1] < 0 or pos[1] >= SCREEN_HEIGHT:
            return "Out of Bounds"

        # check the color of the pixel at the position to determine the element
        color = self.course_surface.get_at(pos)
        
        if color == GREEN_FAIRWAY:
            return "Fairway"
        elif color == GREEN_ROUGH:
            return "Rough"
        elif color == GREEN_TEEBOX:
            return "Teebox"
        elif color == GREEN_GREEN or color == RED or color == BLACK:
            return "Green"
        elif color == YELLOW:
            return "Bunker"
        elif color == BLUE:
            return "Water Hazard"
        else:
            return "Out of Bounds"