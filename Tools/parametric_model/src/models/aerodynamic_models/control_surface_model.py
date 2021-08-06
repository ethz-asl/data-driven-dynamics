__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np

from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation
from progress.bar import Bar

"""
The control surface model is conform to PX4's standard plane
 """


class ControlSurfaceModel():
    def __init__(self, config_dict, actuator_input_vec, stall_angle=35.0, sig_scale_fac=30):
        self.stall_angle = stall_angle*math.pi/180.0
        self.sig_scale_fac = sig_scale_fac
        self.name = config_dict["description"]
        self.actuator_input_vec = np.array(actuator_input_vec)
        self.n_timestamps = actuator_input_vec.shape[0]
        self.air_density = 1.225

    def compute_actuator_force_features(self, index, v_airspeed, angle_of_attack):
        """
        Model description:
        """
        actuator_input = self.actuator_input_vec[index]
        q_xz = 0.5 * self.air_density * (v_airspeed[0]**2 + v_airspeed[2]**2) #TODO Take dynamic pressure
        # TODO Compute lift axis and drag axis
        lift_axis = np.array([v_airspeed[0], 0.0, v_airspeed[2]])
        lift_axis = (lift_axis / np.linalg.norm(lift_axis)).reshape((3, 1))
        drag_axis = (-1.0 * v_airspeed / np.linalg.norm(v_airspeed)).reshape((3, 1))
        X_lift = lift_axis @ np.array([[actuator_input]]) *q_xz
        X_drag = drag_axis @ np.array([[actuator_input]]) *q_xz
        R_aero_to_body = Rotation.from_rotvec([0, -angle_of_attack, 0]).as_matrix()
        X_lift_body = R_aero_to_body @ X_lift
        X_drag_body = R_aero_to_body @ X_drag

        return np.hstack((X_lift_body, X_drag_body))

    def compute_actuator_force_matrix(self, v_airspeed_mat, angle_of_attack_vec):
        print("Computing force features for rotor:", self.name)

        X_forces = self.compute_actuator_force_features(0, v_airspeed_mat[0, :], angle_of_attack_vec[0, :])
        rotor_features_bar = Bar(
            'Feature Computatiuon', max=self.actuator_input_vec.shape[0])
        for index in range(1, self.n_timestamps):
            X_force_curr = self.compute_actuator_force_features(index, v_airspeed_mat[index, :], angle_of_attack_vec[index, :])
            X_forces = np.vstack((X_forces, X_force_curr))
            rotor_features_bar.next()
        rotor_features_bar.finish()
        coef_list_forces = ["c_l_delta", "c_d_delta"]
        self.X_forces = X_forces
        self.X_thrust = X_forces[:, 1:]
        return X_forces, coef_list_forces
