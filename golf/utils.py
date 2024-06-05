import numpy as np
from constants import WIDTH


def bezier_curve(p0, p1, p2, p3, t):
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3

def generate_bezier_path(points, num_points=1000):
    path = []
    for i in range(len(points) - 1):
        p0 = points[i]
        p3 = points[i + 1]
        tangent = points[i + 1] - points[i - 1] if i > 0 else points[i + 1] - points[i]
        p1 = p0 + tangent / 3
        tangent = points[i + 2] - points[i] if i < len(points) - 2 else points[i + 1] - points[i]
        p2 = p3 - tangent / 3
        for t in np.linspace(0, 1, num_points // (len(points) - 1)):
            path.append(bezier_curve(p0, p1, p2, p3, t))

    # purge duplicate points
    path = np.array(path)
    path = np.unique(path, axis=0)

    return path

def generate_height_envelope(point, left_bound, right_bound):
    # Calculate the height of the envelope at a given x-coordinate
    # the envelop function is between 0 and 1, so we need to normalize the x values to be between 0 and 1
    x, y = point
    normalized_x = (x - left_bound[0]) / (right_bound[0] - left_bound[0])

    if normalized_x < 0  or normalized_x > 1:
        raise ValueError("The x-coordinate must be between the left and right bounds. The x coordinate was {}".format(x))

    delta_y = (-np.power(2 * normalized_x - 1, 10) + 0.15 * np.sin(7 * (3 * normalized_x - 1)) + 1) * WIDTH

    if delta_y < 0:
        delta_y = 0

    return delta_y