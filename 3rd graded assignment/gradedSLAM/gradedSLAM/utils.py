import numpy as np

#sverre: maps an angle to [-pi, pi)
def wrapToPi(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


#sverre: 2x2 rotation matrix from BODy to WORLD
def rotmat2d(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle),  np.cos(angle)]])
