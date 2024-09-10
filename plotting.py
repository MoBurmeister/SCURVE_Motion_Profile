import matplotlib.pyplot as plt
import numpy as np
from constants import ACCELERATION_ID, SPEED_ID, POSITION_ID

def plot_trajectory(traj, dt):
    """
    Plots the acceleration, speed, and position profiles of a given trajectory.
    Parameters:
    traj (Trajectory): The trajectory object containing the profiles.
    dt (float): The time step for the profiles.
    Returns:
    None
    """
    dof = traj.dof
    timesteps = int(max(traj.time) / dt)
    time = np.linspace(0, max(traj.time), timesteps)

    p_list = [traj(t) for t in time]
    profiles = np.asarray(p_list)

    r_profiles = np.zeros((dof, 3, timesteps))
    for d in range(dof):
        for p in range(3):
            r_profiles[d, p, :] = profiles[:, d, p]

    fig = plt.figure(0)

    for i, profile in zip(range(dof), r_profiles):
        plt.subplot(300 + dof * 10 + (i + 1))
        plt.title("Acceleration profile")
        plt.plot(time, profile[ACCELERATION_ID][:])

        plt.subplot(300 + dof * 10 + (i + 1) + dof)
        plt.title("Speed profile")
        plt.plot(time, profile[SPEED_ID][:])

        plt.subplot(300 + dof * 10 + (i + 1) + dof * 2)
        plt.title("Position profile")
        plt.plot(time, profile[POSITION_ID][:])

    plt.tight_layout()
    plt.show()