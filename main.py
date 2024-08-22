import matplotlib.pyplot as plt
from scurve_calculation import SCurveMotion


# Define the maximum constraints
max_velocity = 2.0  # m/s
max_acceleration = 1.0  # m/s^2
max_jerk = 2.0  # m/s^3

# Create an instance of SCurveMotion
s_curve = SCurveMotion(max_velocity, max_acceleration, max_jerk)

# Define the specific jerk and acceleration/deceleration values for the motion profile
accel_jerk = 1.5  # m/s^3
accel = 0.8  # m/s^2
decel_jerk = 1.5  # m/s^3
decel = 0.8  # m/s^2
distance = 10.0  # meters

# Calculate the motion profile
motion_profile = s_curve.calculate_motion_profile(accel_jerk, accel, decel_jerk, decel, distance)

# Extract the data for plotting
times = sorted(motion_profile.keys())
positions = [motion_profile[t][0] for t in times]
velocities = [motion_profile[t][1] for t in times]
accelerations = [motion_profile[t][2] for t in times]

# Plot position, velocity, and acceleration over time
plt.figure(figsize=(12, 8))

# Position plot
plt.subplot(3, 1, 1)
plt.plot(times, positions, label="Position (m)")
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()

# Velocity plot
plt.subplot(3, 1, 2)
plt.plot(times, velocities, label="Velocity (m/s)", color='orange')
plt.ylabel("Velocity (m/s)")
plt.grid(True)
plt.legend()

# Acceleration plot
plt.subplot(3, 1, 3)
plt.plot(times, accelerations, label="Acceleration (m/s^2)", color='green')
plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s^2)")
plt.grid(True)
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()
