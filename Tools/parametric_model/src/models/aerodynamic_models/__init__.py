"""Provides interface to access all modules available in the model directly"""

__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from . import aero_model_AAE
from .aero_model_AAE import AeroModelAAE
from . import aero_model_Delta
from .aero_model_Delta import AeroModelDelta
from . import simple_drag_model
from .simple_drag_model import SimpleDragModel
from . import tilt_wing_section_aero_model
from .tilt_wing_section_aero_model import TiltWingSection
