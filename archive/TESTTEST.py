import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod


ACCELERATION_ID = 0
SPEED_ID = 1
POSITION_ID = 2

OPTIMIZER_THRESHOLD = 0.01
EPSILON = 0.0001


class PlanningError(Exception):
    def __init__(self, msg):
        super(Exception, self).__init__(msg)


class Trajectory(object):
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


def plot_trajectory(traj, dt):
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


class TrajectoryPlanner(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def plan_trajectory(self):
        pass

    def _check_shape(self, *args):
        sh = len(args[0])
        for arg in args:
            if sh != len(arg):
                raise ValueError("All parameters must have the same dimension")
        return (sh,)


class ScurvePlanner(TrajectoryPlanner):
    def __init__(self, debug=False):
        self.s = 1

    def __scurve_check_possibility(self, q0, q1, v0, v1, v_max, a_max, j_max):
        dv = np.abs(v1 - v0)
        dq = np.abs(q1 - q0)

        time_to_reach_max_a = a_max / j_max
        time_to_set_set_speeds = np.sqrt(dv / j_max)

        Tj = min(time_to_reach_max_a, time_to_set_set_speeds)

        if Tj == time_to_reach_max_a:
            return dq > 0.5 * (v0 + v1) * (Tj + dv / a_max)
        elif Tj < time_to_reach_max_a:
            return dq > Tj * (v0 + v1)
        else:
            raise PlanningError("Something went wrong")

    def __compute_maximum_speed_reached(self, q0, q1, v0, v1, v_max, a_max, j_max):
        if (v_max - v0) * j_max < a_max**2:
            Tj1 = np.sqrt((v_max - v0) / j_max)
            Ta = 2 * Tj1
        else:
            Tj1 = a_max / j_max
            Ta = Tj1 + (v_max - v0) / a_max

        if (v_max - v1) * j_max < a_max**2:
            Tj2 = np.sqrt((v_max - v1) / j_max)
            Td = 2 * Tj2
        else:
            Tj2 = a_max / j_max
            Td = Tj2 + (v_max - v1) / a_max

        Tv = (q1 - q0) / v_max - (Ta / 2) * (1 + v0 / v_max) - (Td / 2) * (1 + v1 / v_max)

        if Tv < 0:
            raise PlanningError("Maximum velocity is not reached. Failed to plan trajectory")

        return Tj1, Ta, Tj2, Td, Tv

    def __compute_maximum_speed_not_reached(self, q0, q1, v0, v1, v_max, a_max, j_max):
        Tj1 = Tj2 = Tj = a_max / j_max
        Tv = 0

        v = (a_max**2) / j_max
        delta = ((a_max**4) / (j_max**2)) + 2 * ((v0**2) + (v1**2)) + a_max * (4 * (q1 - q0) - 2 * (a_max / j_max) * (v0 + v1))

        Ta = (v - 2 * v0 + np.sqrt(delta)) / (2 * a_max)
        Td = (v - 2 * v1 + np.sqrt(delta)) / (2 * a_max)

        if (Ta - 2 * Tj < EPSILON) or (Td - 2 * Tj < EPSILON):
            raise PlanningError("Maximum acceleration is not reached. Failed to plan trajectory")

        return Tj1, Ta, Tj2, Td, Tv

    def __scurve_search_planning(self, q0, q1, v0, v1, v_max, a_max, j_max, l=0.99, max_iter=2000, dt_thresh=0.01, T=None):
        _a_max = a_max
        it = 0

        while (it < max_iter) and (_a_max > EPSILON):
            try:
                Tj1, Ta, Tj2, Td, Tv = self.__compute_maximum_speed_not_reached(q0, q1, v0, v1, v_max, _a_max, j_max)

                if T is None:
                    return Tj1, Ta, Tj2, Td, Tv

                if abs(T - Ta - Td - Tv) <= dt_thresh:
                    return Tj1, Ta, Tj2, Td, Tv
                else:
                    _a_max *= l
                    it += 1

            except PlanningError:
                it += 1
                _a_max *= l

        raise PlanningError("Failed to find appropriate a_max")

    def __sign_transforms(self, q0, q1, v0, v1, v_max, a_max, j_max):
        v_min = -v_max
        a_min = -a_max
        j_min = -j_max

        s = np.sign(q1 - q0)
        vs1 = (s + 1) / 2
        vs2 = (s - 1) / 2

        _q0 = s * q0
        _q1 = s * q1
        _v0 = s * v0
        _v1 = s * v1
        _v_max = vs1 * v_max + vs2 * v_min
        _a_max = vs1 * a_max + vs2 * a_min
        _j_max = vs1 * j_max + vs2 * j_min

        return _q0, _q1, _v0, _v1, _v_max, _a_max, _j_max

    def __point_sign_transform(self, q0, q1, p):
        s = np.sign(q1 - q0)
        return s * p

    def __get_trajectory_func(self, Tj1, Ta, Tj2, Td, Tv, q0, q1, v0, v1, v_max, a_max, j_max):
        T = Ta + Td + Tv
        a_lim_a = j_max * Tj1
        a_lim_d = -j_max * Tj2
        v_lim = v0 + (Ta - Tj1) * a_lim_a

        def trajectory(t):
            if 0 <= t < Tj1:
                a = j_max * t
                v = v0 + j_max * (t**2) / 2
                q = q0 + v0 * t + j_max * (t**3) / 6
            elif Tj1 <= t < Ta - Tj1:
                a = a_lim_a
                v = v0 + a_lim_a * (t - Tj1 / 2)
                q = q0 + v0 * t + a_lim_a * (3 * (t**2) - 3 * Tj1 * t + Tj1**2) / 6
            elif Ta - Tj1 <= t < Ta:
                tt = Ta - t
                a = j_max * tt
                v = v_lim - j_max * (tt**2) / 2
                q = q0 + (v_lim + v0) * Ta / 2 - v_lim * tt + j_max * (tt**3) / 6
            elif Ta <= t < Ta + Tv:
                a = 0
                v = v_lim
                q = q0 + (v_lim + v0) * Ta / 2 + v_lim * (t - Ta)
            elif T - Td <= t < T - Td + Tj2:
                tt = t - T + Td
                a = -j_max * tt
                v = v_lim - j_max * (tt**2) / 2
                q = q1 - (v_lim + v1) * Td / 2 + v_lim * tt - j_max * (tt**3) / 6
            elif T - Td + Tj2 <= t < T - Tj2:
                tt = t - T + Td
                a = a_lim_d
                v = v_lim + a_lim_d * (tt - Tj2 / 2)
                q = q1 - (v_lim + v1) * Td / 2 + v_lim * tt + a_lim_d * (3 * (tt**2) - 3 * Tj2 * tt + Tj2**2) / 6
            elif T - Tj2 <= t < T:
                tt = T - t
                a = -j_max * tt
                v = v1 + j_max * (tt**2) / 2
                q = q1 - v1 * tt - j_max * (tt**3) / 6
            else:
                a = 0
                v = v1
                q = q1

            point = np.zeros((3,), dtype=np.float32)
            point[ACCELERATION_ID] = a
            point[SPEED_ID] = v
            point[POSITION_ID] = q

            return point

        return trajectory

    def __get_trajectory_function(self, q0, q1, v0, v1, v_max, a_max, j_max, Tj1, Ta, Tj2, Td, Tv):
        zipped_args = self.__sign_transforms(q0, q1, v0, v1, v_max, a_max, j_max)
        traj_func = self.__get_trajectory_func(Tj1, Ta, Tj2, Td, Tv, *zipped_args)

        def sign_back_transformed(t):
            return self.__point_sign_transform(q0, q1, traj_func(t))

        return sign_back_transformed

    def __scurve_profile_no_opt(self, q0, q1, v0, v1, v_max, a_max, j_max):
        if self.__scurve_check_possibility(q0, q1, v0, v1, v_max, a_max, j_max):
            try:
                Tj1, Ta, Tj2, Td, Tv = self.__compute_maximum_speed_reached(q0, q1, v0, v1, v_max, a_max, j_max)
            except PlanningError:
                try:
                    Tj1, Ta, Tj2, Td, Tv = self.__compute_maximum_speed_not_reached(q0, q1, v0, v1, v_max, a_max, j_max)
                except PlanningError:
                    try:
                        Tj1, Ta, Tj2, Td, Tv = self.__scurve_search_planning(q0, q1, v0, v1, v_max, a_max, j_max)
                    except PlanningError:
                        raise PlanningError("Trajectory is infeasible")
            return np.asarray([Tj1, Ta, Tj2, Td, Tv], dtype=np.float32)
        else:
            raise PlanningError("Trajectory is not feasible")

    def __put_params(self, params_list, params, dof):
        for i in range(len(params_list)):
            params_list[i][dof] = params[i]

    def __get_dof_time(self, params_list, dof):
        return params_list[1][dof] + params_list[3][dof] + params_list[4][dof]

    def __get_traj_params_containers(self, sh):
        T = np.zeros(sh)
        Ta = np.zeros(sh)
        Tj1 = np.zeros(sh)
        Td = np.zeros(sh)
        Tj2 = np.zeros(sh)
        Tv = np.zeros(sh)
        return T, Tj1, Ta, Tj2, Td, Tv

    def __plan_trajectory_1D(self, q0, q1, v0, v1, v_max, a_max, j_max, T=None):
        zipped_args = self.__sign_transforms(q0, q1, v0, v1, v_max, a_max, j_max)

        if T is None:
            res = self.__scurve_profile_no_opt(*zipped_args)
        else:
            res = self.__scurve_search_planning(*zipped_args, T=T)

        T = res[1] + res[3] + res[4]
        a_max_c = res[0] * j_max
        a_min_c = a_max_c - res[2] * j_max

        return res

    def plan_trajectory(self, q0, q1, v0, v1, v_max, a_max, j_max, t=None):
        sh = self._check_shape(q0, q1, v0, v1)
        ndof = sh[0]

        task_list = np.asarray([q0, q1, v0, v1, [v_max] * ndof, [a_max] * ndof, [j_max] * ndof], dtype=np.float32)
        T, Tj1, Ta, Tj2, Td, Tv = self.__get_traj_params_containers(sh)
        trajectory_params = np.asarray([Tj1, Ta, Tj2, Td, Tv], dtype=np.float32)
        trajectory_funcs = []

        dq = np.subtract(q1, q0)
        max_displacement_id = np.argmax(np.abs(dq))

        max_displacement_params = self.__plan_trajectory_1D(*task_list[:, max_displacement_id], T=t)

        self.__put_params(trajectory_params, max_displacement_params, max_displacement_id)
        max_displacement_time = self.__get_dof_time(trajectory_params, max_displacement_id)
        T[max_displacement_id] = max_displacement_time

        for _q0, _q1, _v0, _v1, ii in zip(q0, q1, v0, v1, range(ndof)):
            if ii == max_displacement_id:
                continue

            if _v1 != 0:
                traj_params = self.__plan_trajectory_1D(_q0, _q1, _v0, _v1, v_max, a_max, j_max, T=max_displacement_time)
            else:
                traj_params = self.__plan_trajectory_1D(_q0, _q1, _v0, _v1, v_max, a_max, j_max)

            T[ii] = Ta[ii] + Td[ii] + Tv[ii]
            self.__put_params(trajectory_params, traj_params, ii)

        for dof in range(ndof):
            tr_func = self.__get_trajectory_function(q0[dof], q1[dof], v0[dof], v1[dof], v_max, a_max, j_max, *trajectory_params[:, dof])
            trajectory_funcs.append(tr_func)

        tr = Trajectory()
        tr.time = (T[max_displacement_id],)
        tr.trajectory = trajectory_funcs
        tr.dof = ndof

        return tr


import numpy as np

# Create an instance of the S-curve planner
planner = ScurvePlanner()

# Define the start and end positions (q0 and q1), initial and final velocities (v0 and v1)
# Example: planning a 1-DOF trajectory (one dimensional) with given velocity and acceleration limits
q0 = np.array([0.0])  # initial position
q1 = np.array([10.0])  # final position
v0 = np.array([0.0])  # initial velocity
v1 = np.array([0.0])  # final velocity

# Define the velocity, acceleration, and jerk limits
v_max = 10.0  # maximum velocity
a_max = 2.0  # maximum acceleration
j_max = 1  # maximum jerk

# Plan the trajectory
trajectory = planner.plan_trajectory(q0, q1, v0, v1, v_max, a_max, j_max)

# Time step for sampling the trajectory (for plotting)
dt = 0.01

# Plot the planned trajectory (position, velocity, and acceleration profiles)
plot_trajectory(trajectory, dt)
