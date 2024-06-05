import numpy as np
import scipy.stats
import pygame

SCREEN_WIDTH, SCREEN_HEIGHT = 1200, 800
DECAY_RATE = 0.95

class AimingSystem:
    def __init__(self, params):
        self.params = params
        self.current_club = 'Driver'
        self.current_lie = 'Teebox'
        self.trail_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        self.prev_target = None

    def draw_arrow(self, screen, ball_pos, target_pos):
        direction = np.array(target_pos) - ball_pos
        magnitude = np.linalg.norm(direction)
        unit_vector = direction / magnitude
        end_pos = ball_pos + unit_vector * min(magnitude, 100, self.params[self.current_club][self.current_lie]["distance"])
        pygame.draw.line(screen, (0, 0, 0), ball_pos, end_pos, 3)

        if magnitude > 10:
            arrowhead = end_pos - 10 * unit_vector
            arrowhead_perp = np.array([unit_vector[1], -unit_vector[0]]) * 5
            pygame.draw.polygon(screen, (0, 0, 0), [end_pos, arrowhead + arrowhead_perp, arrowhead - arrowhead_perp])

    def draw_gaussian(self, screen, ball_pos, target_pos):
        direction = np.array(target_pos) - ball_pos
        distance = self.params[self.current_club][self.current_lie]["distance"]
        mean_pos = ball_pos + distance * (direction / np.linalg.norm(direction))

        mean = [mean_pos[0], mean_pos[1]]
        cov = self._get_cov_matrix(self.current_club, self.current_lie, direction)

        self._draw_gaussian_distribution(screen, mean, cov)

    def _get_cov_matrix(self, club, lie, direction):
        horizontal_std = self.params[club][lie]["horizontal_std"]
        vertical_std = self.params[club][lie]["vertical_std"]

        unrotated_cov = np.array([[horizontal_std**2, 0], [0, vertical_std**2]])

        angle = -np.arctan2(direction[1], direction[0])
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        rotated_cov = rotation_matrix @ unrotated_cov @ rotation_matrix.T

        return rotated_cov

    def _draw_gaussian_distribution(self, screen, mean, cov):
        size = 7 * int(np.sqrt(np.max(cov))) # Define the size of the surface

        center = size // 2

        # Create a surface to draw the Gaussian distribution
        gaussian_surface = pygame.Surface((size, size), pygame.SRCALPHA)

        # Generate the grid of points
        x, y = np.meshgrid(np.linspace(-center, center, size), np.linspace(-center, center, size))
        pos = np.dstack((x, y))
        
        # Generate the Gaussian distribution
        rv = scipy.stats.multivariate_normal([0, 0], cov)
        pdf_values = rv.pdf(pos)
        pdf_values = (pdf_values / pdf_values.max() * 255).astype(np.uint8)


        # Create the RGB array and the Alpha channel array
        rgb_array = np.zeros((size, size, 3), dtype=np.uint8)
        rgb_array[..., 0] = 255  # Set the red channel
        alpha_array = pdf_values

        # Create a surface from the RGB array
        gaussian_surface = pygame.surfarray.make_surface(rgb_array)
        gaussian_surface = gaussian_surface.convert_alpha()

        # Set the alpha channel
        pygame.surfarray.pixels_alpha(gaussian_surface)[:] = alpha_array

        # Blit the Gaussian surface onto the main screen
        pos_x, pos_y = int(mean[0]) - center, int(mean[1]) - center
        screen.blit(gaussian_surface, (pos_x, pos_y))

    def sample_gaussian(self, ball_pos, target_pos):
        self.prev_target = target_pos # store the intended target
        direction = np.array(target_pos) - ball_pos
        distance = self.params[self.current_club][self.current_lie]["distance"]
        mean_pos = ball_pos + distance * (direction / np.linalg.norm(direction))

        mean = [mean_pos[0], mean_pos[1]]
        cov = self._get_cov_matrix(self.current_club, self.current_lie, direction)

        sample = np.random.multivariate_normal(mean, cov)
        return sample

    def change_club(self, club):
        self.current_club = club

    def set_lie(self, lie):
        self.current_lie = lie
