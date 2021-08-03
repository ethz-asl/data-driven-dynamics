"""Provides interface to access all modules available in the model directly"""

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


from .aero_model_AAE import AeroModelAAE
from .aero_model_Delta import AeroModelDelta
from .fuselage_drag_model import FuselageDragModel
from .elevator_model import ElevatorModel
from .standard_wing_model import StandardWingModel
