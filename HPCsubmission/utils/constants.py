"""
Constants are defined to be used across the module.
"""

import numpy as np

# Configuration
A = 2
Q = 9

# Boundaries
LEFT = "left"
RIGHT = "right"
TOP = "top"
BOTTOM = "bottom"

# Density weight contribution along the q different directions
W_I = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])

# Velocity components
C_AI = np.array([[0, 1, 0, -1, 0, 1, -1, -1, 1],    # Velocity x-component
                 [0, 0, 1, 0, -1, 1, 1, -1, -1]])   # Velocity y-component

# Varying parameter for Reynolds number
VELOCITY = 'velocity'
OMEGA = 'omega'