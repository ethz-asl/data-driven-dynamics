__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np

from ...tools import symmetric_logistic_sigmoid
from scipy.spatial.transform import Rotation


class LinearPlateAeroModel():
    def __init__(self, stall_angle=20.0):
        self.stall_angle = stall_angle

    def compute_single_aero_feature(self, v_airspeed, angle_of_attack):
        # compute lift and drag forces in aero_frame, where x is orented along F_drag and z along F_lift.
        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        F_xz_aero_frame = np.zeros((3, 4))
        F_xz_aero_frame[0, 3] = v_xz**2
        F_xz_aero_frame[2, 0] = (
            1 - symmetric_logistic_sigmoid(angle_of_attack, self.stall_angle))*angle_of_attack*v_xz**2
        F_xz_aero_frame[2, 1] = (
            1 - symmetric_logistic_sigmoid(angle_of_attack, self.stall_angle))*v_xz**2
        F_xz_aero_frame[2, 2] = 2 * \
            symmetric_logistic_sigmoid(angle_of_attack, self.stall_angle) \
            * math.sin(angle_of_attack)*math.cos(angle_of_attack)*v_xz**2

        # Transorm from aero frame to body FRD frame
        R_aero_to_body = -1 * \
            Rotation.from_rotvec([0, angle_of_attack, 0]).as_matrix()
        F_xz_body_frame = R_aero_to_body @ F_xz_aero_frame
        # compute drag in y direction of body frame
        F_y_body_frame = np.array([0, v_airspeed[1]**2, 0]).reshape(3, 1)
        X_aero = np.hstack((F_xz_body_frame, F_y_body_frame))
        return X_aero

    def compute_aero_features(self, v_airspeed_mat, angle_of_attack_vec):
        """inputs:
        v_airspeed_mat: matrice with columns for [ax, ay, az]
        angle_of_attack_vec: vector with corresponding AoA values"""
        X_aero = self.compute_single_aero_feature(
            v_airspeed_mat[0, :], angle_of_attack_vec[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_single_aero_feature(
                v_airspeed_mat[i, :], angle_of_attack_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
        return X_aero


if __name__ == "__main__":
    linearPlateAeroModel = LinearPlateAeroModel(20.0)
