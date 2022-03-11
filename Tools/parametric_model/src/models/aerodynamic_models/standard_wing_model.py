"""
 *
 * Copyright (c) 2021 Manuel Yves Galliker
 *               2021 Autonomous Systems Lab ETH Zurich
 * All rights reserved.
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name Data Driven Dynamics nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
"""

__author__ = "Manuel Yves Galliker"
__maintainer__ = "Manuel Yves Galliker"
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

        # compute dynamic pressure times wing area
        q_x_A = 0.5 * self.air_density * self.area * \
            (v_airspeed[0]**2 + v_airspeed[2]**2)  # TODO Take dynamic pressure
        X_wing_aero_frame = np.zeros((3, 8))

        # region interpolation using a symmetric sigmoid function
        # 0 in linear/quadratic region, 1 in post-stall region

        stall_region = cropped_sym_sigmoid(
            angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region

        # Compute Drag force coeffiecients:
        X_wing_aero_frame[0, 0] = -flow_attached_region * q_x_A
        X_wing_aero_frame[0, 1] = - \
            flow_attached_region * q_x_A * angle_of_attack
        X_wing_aero_frame[0, 2] = -flow_attached_region * \
            q_x_A * angle_of_attack**2
        X_wing_aero_frame[0, 3] = -stall_region * \
            (1 - math.sin(angle_of_attack)**2)*q_x_A
        X_wing_aero_frame[0, 4] = -stall_region * \
            (math.sin(angle_of_attack)**2)*q_x_A
        # Compute Lift force coefficients:
        X_wing_aero_frame[2, 5] = -flow_attached_region*q_x_A
        X_wing_aero_frame[2, 6] = -flow_attached_region*q_x_A*angle_of_attack
        X_wing_aero_frame[2, 7] = -stall_region * \
            q_x_A * math.sin(2*angle_of_attack)

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame
        X_wing_body_frame = X_wing_body_frame.flatten()
        return X_wing_body_frame

    def compute_wing_moment_features(self, v_airspeed, angle_of_attack, angle_of_sideslip):
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
        q_xz = 0.5 * self.air_density * \
            (v_airspeed[0]**2 + v_airspeed[2]**2)  # TODO Take dynamic pressure
        q_xy = 0.5 * self.air_density * (v_airspeed[0]**2 + v_airspeed[1]**2)

        X_wing_aero_frame = np.zeros((3, 3))

        # region interpolation using a symmetric sigmoid function
        # 0 in linear/quadratic region, 1 in post-stall region
        stall_region = cropped_sym_sigmoid(
            angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac)
        # 1 in linear/quadratic region, 0 in post-stall region
        flow_attached_region = 1 - stall_region

        # Compute Roll Moment coeffiecients:

        # Compute Pitch Moment coeffiecients:
        X_wing_aero_frame[1, 0] = flow_attached_region * q_xz * self.area
        X_wing_aero_frame[1, 1] = flow_attached_region * \
            q_xz * self.area * angle_of_attack

        # TODO: Compute Yaw Moment coeffiecients:
        X_wing_aero_frame[2, 2] = flow_attached_region * \
            q_xy * self.area * angle_of_sideslip

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame
        X_wing_body_frame = X_wing_body_frame.flatten()
        return X_wing_body_frame

    def compute_aero_force_features(self, v_airspeed_mat, angle_of_attack_vec):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        """
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
        coef_dict = {
            "c_d_wing_xz_offset": {"lin":{ "x": "c_d_wing_xz_offset_x","y": "c_d_wing_xz_offset_y","z":"c_d_wing_xz_offset_z"}},
            "c_d_wing_xz_lin": {"lin":{ "x": "c_d_wing_xz_lin_x","y": "c_d_wing_xz_lin_y","z":"c_d_wing_xz_lin_z"}},
            "c_d_wing_xz_quad": {"lin":{ "x": "c_d_wing_xz_quad_x","y": "c_d_wing_xz_quad_y","z":"c_d_wing_xz_quad_z"}},
            "c_d_wing_xz_fp_min": {"lin":{ "x": "c_d_wing_xz_fp_min_x","y": "c_d_wing_xz_fp_min_y","z":"c_d_wing_xz_fp_min_z"}},
            "c_d_wing_xz_fp_max": {"lin":{ "x": "c_d_wing_xz_fp_max_x","y": "c_d_wing_xz_fp_max_y","z":"c_d_wing_xz_fp_max_z"}},
            "c_l_wing_xz_offset": {"lin":{ "x": "c_l_wing_xz_offset_x","y": "c_l_wing_xz_offset_y","z":"c_l_wing_xz_offset_z"}},
            "c_l_wing_xz_lin": {"lin":{ "x": "c_l_wing_xz_lin_x","y": "c_l_wing_xz_lin_y","z":"c_l_wing_xz_lin_z"}},
            "c_l_wing_xz_fp": {"lin":{ "x": "c_l_wing_xz_fp_x","y": "c_l_wing_xz_fp_y","z":"c_l_wing_xz_fp_z"}},
        }
        col_names = ["c_d_wing_xz_offset_x", "c_d_wing_xz_offset_y", "c_d_wing_xz_offset_z", 
                    "c_d_wing_xz_lin_x", "c_d_wing_xz_lin_y", "c_d_wing_xz_lin_z",
                    "c_d_wing_xz_quad_x", "c_d_wing_xz_quad_y", "c_d_wing_xz_quad_z",
                    "c_d_wing_xz_fp_min_x", "c_d_wing_xz_fp_min_y", "c_d_wing_xz_fp_min_z",
                    "c_d_wing_xz_fp_max_x", "c_d_wing_xz_fp_max_y", "c_d_wing_xz_fp_max_z",
                    "c_l_wing_xz_offset_x", "c_l_wing_xz_offset_y", "c_l_wing_xz_offset_z",
                    "c_l_wing_xz_lin_x", "c_l_wing_xz_lin_y", "c_l_wing_xz_lin_z", 
                    "c_l_wing_xz_fp_x", "c_l_wing_xz_fp_y", "c_l_wing_xz_fp_z"]
                
        return X_aero, coef_dict, col_names

    def compute_aero_moment_features(self, v_airspeed_mat, angle_of_attack_vec, angle_of_sideslip_vec):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        """
        print("Starting computation of aero moment features...")
        X_aero = self.compute_wing_moment_features(
            v_airspeed_mat[0, :], angle_of_attack_vec[0], angle_of_sideslip_vec[0])
        aero_features_bar = Bar(
            'Feature Computatiuon', max=v_airspeed_mat.shape[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_wing_moment_features(
                v_airspeed_mat[i, :], angle_of_attack_vec[i], angle_of_sideslip_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
            aero_features_bar.next()
        aero_features_bar.finish()
        coef_dict = {
            "c_m_x_wing_xz_offset": {"rot":{ "x": "c_m_x_wing_xz_offset_x","y": "c_m_x_wing_xz_offset_y","z":"c_m_x_wing_xz_offset_z"}},
            "c_m_x_wing_xz_lin": {"rot":{ "x": "c_m_x_wing_xz_lin_x","y": "c_m_x_wing_xz_lin_y","z":"c_m_x_wing_xz_lin_z"}},
            "c_m_z_wing_lin": {"rot":{ "x": "c_m_z_wing_lin_x","y": "c_m_z_wing_lin_y","z":"c_m_z_wing_lin_z"}},
        }
        col_names = ["c_m_x_wing_xz_offset_x", "c_m_x_wing_xz_offset_y", "c_m_x_wing_xz_offset_z", 
                    "c_m_x_wing_xz_lin_x", "c_m_x_wing_xz_lin_y", "c_m_x_wing_xz_lin_z",
                    "c_m_z_wing_lin_x", "c_m_z_wing_lin_y", "c_m_z_wing_lin_z"]
        return X_aero, coef_dict, col_names
