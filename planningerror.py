# class for planning errors when trajectories are not feasible

class PlanningError(Exception):
    """
    Custom exception class for planning errors.

    Args:
        msg (str): The error message.

    Attributes:
        msg (str): The error message.

    """
    def __init__(self, msg):
        super(Exception, self).__init__(msg)