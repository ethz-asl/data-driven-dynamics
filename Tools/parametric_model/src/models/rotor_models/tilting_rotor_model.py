__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from . import ChangingAxisRotorModel
import numpy as np
import pandas as pd
import math
from progress.bar import Bar
from . import RotorModel
from scipy.spatial.transform import Rotation


class TiltingRotorModel(ChangingAxisRotorModel):

    def __init__(self, rotor_config_dict, actuator_input_vec, v_airspeed_mat, tilt_actuator_vec, air_density=1.225, angular_vel_mat=None):
        print(rotor_config_dict)
        self.tilt_axis = np.array(rotor_config_dict["tilt_axis"]).reshape(3, 1)
        self.max_tilt_angle = rotor_config_dict["max_tilt_angle_deg"]*math.pi/180.0
        self.tilt_actuator_vec = np.array(tilt_actuator_vec)
        self.n_timestamps = actuator_input_vec.shape[0]
        self.rotor_axis = np.array(
            rotor_config_dict["rotor_axis"]).reshape(3, 1)
        self.compute_rotor_axis_mat()
        super(TiltingRotorModel, self).__init__(rotor_config_dict, actuator_input_vec,
                                                v_airspeed_mat, air_density=1.225, angular_vel_mat=None)

    def compute_rotor_axis_mat(self):
        self.rotor_axis_mat = np.zeros((self.n_timestamps, 3))
        for i in range(self.n_timestamps):
            # Active vector rotation around tilt axis:
            rotvec = self.tilt_axis.flatten() * self.max_tilt_angle * \
                self.tilt_actuator_vec[i]
            R_active_tilt = Rotation.from_rotvec(
                rotvec).as_matrix()
            curr_axis = (
                R_active_tilt @ self.rotor_axis).flatten()
            self.rotor_axis_mat[i, :] = curr_axis/np.linalg.norm(curr_axis)
