__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np

from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation


class SimpleDragModel():
    def __init__(self, stall_angle=35.0, sig_scale_fac=30):
        # convert to radians
        self.stall_angle = stall_angle*math.pi/180.0
        self.sig_scale_fac = sig_scale_fac

    def compute_fuselage_feature(self, v_airspeed, angle_of_attack):
        """
        Model description:

        Compute lift and drag forces in stability axis frame.

        This is done by interpolating two models: 
        1. More suffisticated Model for abs(AoA) < stall_angle

            - Lift force coefficient as linear function of AoA:
                F_Lift = 0.5 * density * area * V_air_xz^2 * (c_l_0 + c_l_lin*AoA)

            - Drag force coefficient as quadratic function of AoA
                F_Drag = 0.5 * density * area * V_air_xz^2 * (c_d_0 + c_d_lin * AoA + c_d_quad * AoA^2)

        2. Simple plate model for abs(AoA) > stall_angle
                F_Lift = density * area * V_air_xz^2 * cos(AoA) * sin(AoA) * c_l_stall
                F_Drag = 0.5 * density * area * V_air_xz^2 * sin(AoA) * c_d_stall

        The two models are interpolated with a symmetric sigmoid function obtained by multiplying two logistic functions:
            if abs(AoA) < stall_angle: sym_sigmoid(AoA) = 0
            if abs(AoA) > stall_angle: sym_sigmoid(AoA) = 1
        """
        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        X_xz_aero_frame = np.zeros((3, 5))

        # region interpolation using a symmetric sigmoid function
        # 0 in linear/quadratic region, 1 in post-stall region
        stall_region = cropped_sym_sigmoid(
            angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region

        # Compute Drag force coeffiecients:
        X_xz_aero_frame[0, 0] = -flow_attached_region*v_xz**2
        X_xz_aero_frame[0, 1] = -flow_attached_region*angle_of_attack*v_xz**2
        X_xz_aero_frame[0, 2] = -flow_attached_region * \
            angle_of_attack**2*v_xz**2
        X_xz_aero_frame[0, 3] = -stall_region * \
            (1 - math.sin(angle_of_attack)**2)*v_xz**2
        X_xz_aero_frame[0, 4] = -stall_region * \
            (math.sin(angle_of_attack)**2)*v_xz**2

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_xz_body_frame = R_aero_to_body @ X_xz_aero_frame

        """
        Compute drag in y direction of body frame using a single coefficient: 
        F_y = 0.5 * density * area * V_air_y^2 * c_d_y"""
        X_y_body_frame = -np.array([0, math.copysign(
            1, v_airspeed[1]) * v_airspeed[1]**2, 0]).reshape(3, 1)
        X_fuselage = np.hstack((X_xz_body_frame, X_y_body_frame))
        return X_fuselage

    def compute_aero_features(self, v_airspeed_mat, angle_of_attack_vec):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        roll_commands = numpy array of dimension (n,3) with columns for [u_roll_1, u_roll_2, u_pitch_collective]
        """
        X_aero = self.compute_fuselage_feature(
            v_airspeed_mat[0, :], angle_of_attack_vec[0])

        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_fuselage_feature(
                v_airspeed_mat[i, :], angle_of_attack_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
        fuselage_coef_list = ["c_d_wing_xy_offset", "c_d_wing_xy_lin", "c_d_wing_xy_quad", "c_d_wing_xz_stall_min", "c_d_wing_xz_stall_90_deg",
                              "c_d_wing_y_offset"]
        return X_aero, fuselage_coef_list
