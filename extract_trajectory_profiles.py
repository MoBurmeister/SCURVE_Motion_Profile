from constants import ACCELERATION_ID, SPEED_ID, POSITION_ID
import numpy as np

def extract_trajectory_profiles(traj, dt):
    """
    Extracts acceleration, speed, and position profiles from a given trajectory.
    Parameters:
    - traj: Trajectory object
        The trajectory object containing the motion profiles.
    - dt: float
        The time step for sampling the trajectory.
    Returns:
    - time: numpy.ndarray
        The time array.
    - acceleration_profile: numpy.ndarray
        The acceleration profile array.
    - speed_profile: numpy.ndarray
        The speed profile array.
    - position_profile: numpy.ndarray
        The position profile array.
    """
    dof = traj.dof
    timesteps = int(max(traj.time) / dt)
    time = np.linspace(0, max(traj.time), timesteps)

    # Initialize lists to hold the acceleration, speed, and position profiles
    acceleration_profile = []
    speed_profile = []
    position_profile = []

    # Sample the trajectory at each time step
    for t in time:
        point = traj(t)  # Get the point (acceleration, speed, position) for each DoF at time t
        # Assuming a single DoF trajectory for simplicity
        acceleration_profile.append([t, point[0, ACCELERATION_ID]])
        speed_profile.append([t, point[0, SPEED_ID]])
        position_profile.append([t, point[0, POSITION_ID]])

    # Convert the profiles to NumPy arrays
    acceleration_profile = np.array(acceleration_profile)
    speed_profile = np.array(speed_profile)
    position_profile = np.array(position_profile)

    # Return the time array and the 2D arrays with time and corresponding values
    return time, acceleration_profile, speed_profile, position_profile
