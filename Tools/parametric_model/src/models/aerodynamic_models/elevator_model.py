__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import math
from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation

"""
This elevator model currently only estimates elevator lift.
"""


class ElevatorModel():
    def __init__(self, air_density=1.225):
        self.air_density = air_density
        self.wing_surface = 0.55
        self.stall_angle = 15*math.pi/180.0
        self.sig_scale_fac = 30
        self.elevator_coef_list = ["elevator_c_l"]

    def compute_single_elevator_feature(self, v_airspeed, u_elev, angle_of_attack):

        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        X_elevator_aero = np.zeros((3, 1))
        qs = 0.5 * self.air_density*self.wing_surface*v_xz**2

        # specific to aero mini
        elevator_angle = -3.781*u_elev**3 + 17.176*u_elev**2 - 36.426*u_elev + 7.659
        # region interpolation using a symmetric sigmoid function
        # 0 in linear/quadratic region, 1 in post-stall region
        stall_region = cropped_sym_sigmoid(
            angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region

        # elevator lift
        X_elevator_aero[2, 0] = flow_attached_region * elevator_angle*qs

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, (-angle_of_attack), 0]).as_matrix()
        X_elevator_body = R_aero_to_body @ X_elevator_aero
        return X_elevator_body

    def compute_elevator_features(self, v_airspeed_mat, elevator_input_vec, angle_of_attack_vec):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        roll_commands = numpy array of dimension (n,3) with columns for [u_roll_1, u_roll_2, u_pitch_collective]
        """
        X_aero = self.compute_single_elevator_feature(
            v_airspeed_mat[0, :], elevator_input_vec[0], angle_of_attack_vec[0])

        for i in range(1, v_airspeed_mat.shape[0]):
            X_curr = self.compute_single_elevator_feature(
                v_airspeed_mat[i, :], elevator_input_vec[i], angle_of_attack_vec[i])
            X_aero = np.vstack((X_aero, X_curr))

        return X_aero, self.elevator_coef_list
