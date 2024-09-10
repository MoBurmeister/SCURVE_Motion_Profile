from abc import ABCMeta, abstractmethod

class TrajectoryPlanner(object):
    """
    Abstract base class for trajectory planners.
    This class defines the interface for trajectory planners and provides a method for planning trajectories.
    Attributes:
        None
    Methods:
        plan_trajectory: Abstract method for planning a trajectory.
    """

    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def plan_trajectory(self):
        pass

    def _check_shape(self, *args):
        """
        Check the shape of the given parameters.

        Args:
            *args: Variable number of arguments representing the parameters.

        Returns:
            tuple: A tuple containing the shape of the parameters.

        Raises:
            ValueError: If the parameters have different dimensions.

        """
        sh = len(args[0])
        for arg in args:
            if sh != len(arg):
                raise ValueError("All parameters must have the same dimension")
        return (sh,)