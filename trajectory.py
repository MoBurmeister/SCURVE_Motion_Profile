import numpy as np

class Trajectory(object):
    """
    A class representing a trajectory.
    Attributes:
        debug (bool): A flag indicating whether debug mode is enabled.
        time (float): The current time of the trajectory.
        dof (int): The number of degrees of freedom.
        trajectory (list): The list of trajectory functions for each degree of freedom.
    Methods:
        __call__(time): Evaluates the trajectory at a given time and returns the corresponding point.
    """
    def __init__(self, debug=True):
        self._debug = debug
        self._trajectory = None
        self._time = 0
        self._dof = 0
        self._p_logged = 0

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, v):
        self._debug = v

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, v):
        self._time = v

    @property
    def dof(self):
        return self._dof

    @dof.setter
    def dof(self, v):
        self._dof = v

    @property
    def trajectory(self):
        return self._trajectory

    @trajectory.setter
    def trajectory(self, v):
        self._trajectory = v

    def __call__(self, time):
        point = np.zeros((self.dof, 3), dtype=np.float32)
        for t, dof in zip(self.trajectory, range(self.dof)):
            dof_point = t(time)
            np.put(point[dof], range(3), dof_point)
        return point