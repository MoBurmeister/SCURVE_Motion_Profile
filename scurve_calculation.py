import math

class SCurveMotion:
    def __init__(self, max_velocity, max_acceleration, max_jerk):
        """
        Initialize the S-curve motion profile with maximum constraints.

        :param max_velocity: Maximum allowed velocity (m/s)
        :param max_acceleration: Maximum allowed acceleration (m/s^2)
        :param max_jerk: Maximum allowed jerk (m/s^3)
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_jerk = max_jerk

    def calculate_motion_profile(self, accel_jerk, accel, decel_jerk, decel, v_max, dist):
        
        #if the distance is > 0, the trajectory is always feasible!

        if dist <= 0:
            raise ValueError("Distance must be greater than zero.")
        
        if v_max * accel_jerk >= accel**2:
            # Case 1a - acc
            T_J_acc = accel / accel_jerk
