__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
import math
from progress.bar import Bar
from . import ChangingAxisRotorModel


class BiDirectionalRotorModel(ChangingAxisRotorModel):

    def __init__(self, rotor_config_dict, actuator_input_vec, v_airspeed_mat, air_density=1.225, angular_vel_mat=None):
        print(rotor_config_dict)
        self.n_timestamps = actuator_input_vec.shape[0]
        self.rotor_axis = np.array(
            rotor_config_dict["rotor_axis"]).reshape(3, 1)
        self.compute_rotor_axis_mat(actuator_input_vec)
        super(BiDirectionalRotorModel, self).__init__(rotor_config_dict, np.absolute(actuator_input_vec),
                                                      v_airspeed_mat, air_density=1.225, angular_vel_mat=None)

    def compute_rotor_axis_mat(self, actuator_input_vec):
        self.rotor_axis_mat = np.zeros((self.n_timestamps, 3))
        for i in range(self.n_timestamps):
            self.rotor_axis_mat[i, :] = self.rotor_axis * \
                np.sign(actuator_input_vec[i])
