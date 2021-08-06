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
    def __init__(self, config_dict, stall_angle=35.0, sig_scale_fac=30):
        self.stall_angle = stall_angle*math.pi/180.0
        self.sig_scale_fac = sig_scale_fac

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
        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        X_wing_aero_frame = np.zeros((3, 8))

        # region interpolation using a symmetric sigmoid function
        # 0 in linear/quadratic region, 1 in post-stall region
        stall_region = cropped_sym_sigmoid(
            angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region

        # Compute Drag force coeffiecients:
        X_wing_aero_frame[0, 0] = -flow_attached_region*v_xz**2
        X_wing_aero_frame[0, 1] = -flow_attached_region*angle_of_attack*v_xz**2
        X_wing_aero_frame[0, 2] = -flow_attached_region * \
            angle_of_attack**2*v_xz**2
        X_wing_aero_frame[0, 3] = -stall_region * \
            (1 - math.sin(angle_of_attack)**2)*v_xz**2
        X_wing_aero_frame[0, 4] = -stall_region * \
            (math.sin(angle_of_attack)**2)*v_xz**2
        # Compute Lift force coefficients:
        X_wing_aero_frame[2, 5] = -flow_attached_region*angle_of_attack*v_xz**2
        X_wing_aero_frame[2, 6] = -flow_attached_region*v_xz**2
        X_wing_aero_frame[2, 7] = -stall_region*v_xz**2

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame

        return X_wing_body_frame

    def compute_control_surface_force_features(self, v_airspeed, angle_of_attack, flap_commands):
        """
        Model description:
        The model assumes that the ailerons will always be 


        flap_commands: TBD
        1: aileron command left/right?
        2. aileron command left/right?
        """

        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        X_cs = np.zeros((3, 2))

        # aileron drag
        X_cs[0, 0] = -(abs(flap_commands[0]) +
                       abs(flap_commands[1]))*v_xz**2
        X_cs[0, 1] = -(abs(flap_commands[0]) +
                       abs(flap_commands[1]))**2*v_xz**2

        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_rudder_body = R_aero_to_body @ X_cs
        return X_rudder_body

    def compute_single_aero_feature(self, v_airspeed, angle_of_attack, flap_commands):
        """
        Used to compute the feature matrices for a single timestamp

        Inputs:

        v_airspeed: numpy array of dimension (1,3) with [v_a_x, v_a_y, v_a_z]
        angle_of_attack: corresponding AoA values
        roll_commands = numpy array of dimension (1,3) with columns for [u_roll_1, u_roll_2, u_pitch_collective]
        """
        X_wing = self.compute_wing_force_features(v_airspeed, angle_of_attack)
        X_cs = self.compute_control_surface_force_features(
            v_airspeed, angle_of_attack, flap_commands)
        return np.hstack((X_wing, X_cs))

    def compute_aero_features(self, v_airspeed_mat, angle_of_attack_vec, flap_commands):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        flap_commands = numpy array of dimension (n,3) with columns for [u_aileron_right, u_aileron_left, u_elevator]
        """
        print("Starting computation of aero features...")
        X_aero = self.compute_single_aero_feature(
            v_airspeed_mat[0, :], angle_of_attack_vec[0], flap_commands[0, :])
        aero_features_bar = Bar(
            'Feature Computatiuon', max=v_airspeed_mat.shape[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_single_aero_feature(
                v_airspeed_mat[i, :], angle_of_attack_vec[i], flap_commands[i, :])
            X_aero = np.vstack((X_aero, X_curr))
            aero_features_bar.next()
        aero_features_bar.finish()
        wing_coef_list = ["c_d_wing_xz_offset", "c_d_wing_xz_lin", "c_d_wing_xz_quad", "c_d_wing_xz_stall_min", "c_d_wing_xz_stall_90_deg",
                          "c_l_wing_xz_offset", "c_l_wing_xz_lin", "c_l_wing_xz_stall"]
        flap_coef_list = ["c_d_ail_lin",
                          "c_d_ail_quad"]
        aero_coef_list = wing_coef_list + flap_coef_list
        return X_aero, aero_coef_list
