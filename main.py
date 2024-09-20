from scurve_planner import ScurvePlanner
from plotting import plot_trajectory
from extract_trajectory_profiles import extract_trajectory_profiles

import numpy as np

# Create an instance of the S-curve planner
planner = ScurvePlanner()

# Define the start and end positions (q0 and q1), initial and final velocities (v0 and v1)
# Example: planning a 1-DOF trajectory (one dimensional) with given velocity and acceleration limits
q0 = np.array([0.0])  # initial position in m
q1 = np.array([100.0])  # final position in m

#final and initial velocity should be zero for the cart control but can be changed here
v0 = np.array([0.0])  # initial velocity in m/s
v1 = np.array([0.0])  # final velocity in m/s

# Define the velocity, acceleration, and jerk limits
v_max = 10.0  # maximum velocity in m/s
a_max = 8.0  # maximum acceleration in m/s^2
j_max = 3  # maximum jerk in m/s^3

# Plan the trajectory
trajectory = planner.plan_trajectory(q0, q1, v0, v1, v_max, a_max, j_max)

print(trajectory)

# Time step for sampling the trajectory (for plotting) - the overall duration of the drive will be splitted into T/dt steps 
dt = 0.01

# Plot the planned trajectory (position, velocity, and acceleration profiles)
plot_trajectory(trajectory, dt)

time, acceleration_profile, speed_profile, position_profile = extract_trajectory_profiles(trajectory, 0.1)

print(acceleration_profile)

print(speed_profile)

print(acceleration_profile.shape)
