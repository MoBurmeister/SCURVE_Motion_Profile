from scurve_planner import ScurvePlanner
from plotting import plot_trajectory
from extract_trajectory_profiles import extract_trajectory_profiles

import numpy as np

# Create an instance of the S-curve planner
planner = ScurvePlanner()

# Define the start and end positions (q0 and q1), initial and final velocities (v0 and v1)
# Example: planning a 1-DOF trajectory (one dimensional) with given velocity and acceleration limits
q0 = np.array([0.0])  # initial position
q1 = np.array([100.0])  # final position
v0 = np.array([0.0])  # initial velocity
v1 = np.array([0.0])  # final velocity

# Define the velocity, acceleration, and jerk limits
v_max = 10.0  # maximum velocity
a_max = 2.0  # maximum acceleration
j_max = 1  # maximum jerk

# Plan the trajectory
trajectory = planner.plan_trajectory(q0, q1, v0, v1, v_max, a_max, j_max)

print(trajectory)

# Time step for sampling the trajectory (for plotting)
dt = 0.01

# Plot the planned trajectory (position, velocity, and acceleration profiles)
plot_trajectory(trajectory, dt)

time, acceleration_profile, speed_profile, position_profile = extract_trajectory_profiles(trajectory, dt)

print(acceleration_profile)

print(acceleration_profile.shape)
