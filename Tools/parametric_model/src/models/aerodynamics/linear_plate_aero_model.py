__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np

from ...tools import sym_sigmoid
from scipy.spatial.transform import Rotation


class LinearPlateAeroModel():
    def __init__(self, stall_angle=20.0):
        self.stall_angle = stall_angle

    def compute_main_wing_feature(self, v_airspeed, angle_of_attack):
        # compute lift and drag forces in stability axis frame.
        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        F_xz_aero_frame = np.zeros((3, 7))
        F_xz_aero_frame[0, 3] = -(
            1 - sym_sigmoid(angle_of_attack, self.stall_angle))*v_xz**2
        F_xz_aero_frame[0, 4] = -(
            1 - sym_sigmoid(angle_of_attack, self.stall_angle))*angle_of_attack*v_xz**2
        F_xz_aero_frame[0, 5] = -(
            1 - sym_sigmoid(angle_of_attack, self.stall_angle))*angle_of_attack**2*v_xz**2
        F_xz_aero_frame[0, 6] = -(sym_sigmoid(angle_of_attack,
                                  self.stall_angle))*math.sin(angle_of_attack)*v_xz**2
        F_xz_aero_frame[2, 0] = -(
            1 - sym_sigmoid(angle_of_attack, self.stall_angle))*angle_of_attack*v_xz**2
        F_xz_aero_frame[2, 1] = -(
            1 - sym_sigmoid(angle_of_attack, self.stall_angle))*v_xz**2
        F_xz_aero_frame[2, 2] = -2 * \
            sym_sigmoid(angle_of_attack, self.stall_angle) \
            * math.sin(angle_of_attack)*math.cos(angle_of_attack)*v_xz**2

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        F_xz_body_frame = R_aero_to_body @ F_xz_aero_frame
        # compute drag in y direction of body frame
        F_y_body_frame = -np.array([0, math.copysign(
            1, v_airspeed[1]) * v_airspeed[1]**2, 0]).reshape(3, 1)
        X_aero = np.hstack((F_xz_body_frame, F_y_body_frame))
        return X_aero

    def compute_aileron_feature(self, v_airspeed, angle_of_attack, flap_commands):
        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        X_rudder_aero = np.zeros((3, 5))
        X_rudder_aero[0, 0] = - (flap_commands[2] - 0.6)*v_xz**2
        X_rudder_aero[0, 1] = - (flap_commands[2] - 0.6)**2*v_xz**2
        X_rudder_aero[0, 2] = v_xz**2
        X_rudder_aero[1, 3] = (flap_commands[0])*v_xz**2
        X_rudder_aero[1, 4] = (flap_commands[1])*v_xz**2
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_rudder_body = R_aero_to_body @ X_rudder_aero
        return X_rudder_body

    def compute_single_aero_feature(self, v_airspeed, angle_of_attack, flap_commands):
        X_wing = self.compute_main_wing_feature(v_airspeed, angle_of_attack)
        X_rudder = self.compute_aileron_feature(
            v_airspeed, angle_of_attack, flap_commands)
        return np.hstack((X_wing, X_rudder))

    def compute_aero_features(self, v_airspeed_mat, angle_of_attack_vec, flap_commands):
        """inputs:
        v_airspeed_mat: numpy array of dimension (n,3) with columns for [ax, ay, az]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        roll_commands = numpy array of dimension (n,2) with columns for [ax, ay, az]
        """
        X_aero = self.compute_single_aero_feature(
            v_airspeed_mat[0, :], angle_of_attack_vec[0], flap_commands[0, :])

        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_single_aero_feature(
                v_airspeed_mat[i, :], angle_of_attack_vec[i], flap_commands[i, :])
            X_aero = np.vstack((X_aero, X_curr))
        return X_aero


if __name__ == "__main__":
    linearPlateAeroModel = LinearPlateAeroModel(20.0)
