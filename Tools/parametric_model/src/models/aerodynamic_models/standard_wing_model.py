__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np

from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation
from progress.bar import Bar

"""
The standard wing model is conform to PX4's standard plane and models
 a wing with ailerons but without flaps. For reference see:
 
https://docs.px4.io/master/en/airframes/airframe_reference.html#standard-plane
 """


class StandardWingModel():
    def __init__(self, config_dict):
        self.stall_angle = config_dict["stall_angle_deg"]*math.pi/180.0
        self.sig_scale_fac = config_dict["sig_scale_factor"]
        self.air_density = 1.225
        self.area = config_dict["area"]

    def compute_wing_force_features(self, v_airspeed, angle_of_attack):
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
            if abs(AoA) < stall_angle: cropped_sym_sigmoid(AoA) = 0
            if abs(AoA) > stall_angle: cropped_sym_sigmoid(AoA) = 1
        """
        q_xz = 0.5 * self.air_density * (v_airspeed[0]**2 + v_airspeed[2]**2) #TODO Take dynamic pressure
        X_wing_aero_frame = np.zeros((3, 5))

        # region interpolation using a symmetric sigmoid function
        # 0 in linear/quadratic region, 1 in post-stall region

        stall_region = cropped_sym_sigmoid(
            angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region

        # Compute Drag force coeffiecients:
        X_wing_aero_frame[0, 0] = -flow_attached_region * q_xz * self.area
        X_wing_aero_frame[0, 1] = -flow_attached_region * q_xz * self.area * angle_of_attack
        X_wing_aero_frame[0, 2] = -flow_attached_region * q_xz * self.area * angle_of_attack**2
        # Compute Lift force coefficients:
        # Stall region: stall_region * 2 * math.sin(angle_of_attack) * math.cos(angle_of_attack)
        X_wing_aero_frame[2, 3] = -flow_attached_region* q_xz  * self.area
        X_wing_aero_frame[2, 4] = -flow_attached_region * q_xz * self.area * angle_of_attack

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame

        return X_wing_body_frame

    def compute_aero_features(self, v_airspeed_mat, angle_of_attack_vec):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        flap_commands = numpy array of dimension (n,3) with columns for [u_aileron_right, u_aileron_left, u_elevator]
        """
        print("Starting computation of aero features...")
        X_aero = self.compute_wing_force_features(
            v_airspeed_mat[0, :], angle_of_attack_vec[0])
        aero_features_bar = Bar(
            'Feature Computatiuon', max=v_airspeed_mat.shape[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_wing_force_features(
                v_airspeed_mat[i, :], angle_of_attack_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
            aero_features_bar.next()
        aero_features_bar.finish()
        wing_coef_list = ["c_d_wing_xz_offset", "c_d_wing_xz_lin", "c_d_wing_xz_quad",
                          "c_l_wing_xz_offset", "c_l_wing_xz_lin"]
        return X_aero, wing_coef_list
