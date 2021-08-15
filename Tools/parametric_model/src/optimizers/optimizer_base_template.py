__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from abc import ABC, abstractmethod
from typing import Dict, List


class ParametersNotEstimatedError(Exception):
    def __init__(self):
        self.message = "Optimization parameters are not yet estimated." + \
            "Did you call estimate_parameters() method?"
        super().__init__(self.message)


class OptimizerBaseTemplate(ABC):

    """ 
    This is an abstract class that is used as the base 
    template to all other optimizer classes by overwriting the 
    abstract methods. 
    """

    def __init__(self, optimizer_config):
        self.estimation_completed = False
        self.parametersNotEstimatedError = ParametersNotEstimatedError()

    def check_estimation_completed(self):
        if self.estimation_completed:
            return
        else:
            raise self.parametersNotEstimatedError

    @abstractmethod
    def estimate_parameters(self) -> None:
        pass

    @abstractmethod
    def get_optimization_parameters(self) -> List:
        pass

    @abstractmethod
    def predict(self) -> List:
        pass

    @abstractmethod
    def compute_optimization_metrics(self) -> Dict:
        pass
