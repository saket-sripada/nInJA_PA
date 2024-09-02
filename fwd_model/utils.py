import numpy as np

from config import L, pitch, width


# Transducer locations for rectangular linear array
def x_i(i):
    # Equation for the x-coordinate of the i-th transducer element
    # This is based on the geometry described in your prompt
    return -L / 2 + pitch * (i) + width / 2


def r0(i):
    # Return the center coordinates of the i-th transducer element
    # This replaces the spherical coordinates used in the original paper
    return np.array([x_i(i), 0, 0])
