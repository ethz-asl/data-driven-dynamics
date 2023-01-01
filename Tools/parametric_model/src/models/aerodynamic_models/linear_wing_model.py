"""
 *
 * Copyright (c) 2021 Julius Schlapbach
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

__author__ = "Julius Schlapbach"
__maintainer__ = "Julius Schlapbach"
__license__ = "BSD 3"

import math
import numpy as np

from scipy.spatial.transform import Rotation
from progress.bar import Bar


class LinearWingModel():
    def __init__(self, config_dict):
        self.air_density = 1.225
        self.gravity = 9.81
        self.area = config_dict["area"]
        self.chord = config_dict["chord"]

    def compute_wing_force_features(self, v_airspeed, angle_of_attack, elevator_input):
        """
        Model description:

        Compute lift and drag forces in stability axis frame.

        Lift force is modeled as a linear function of the angle of attack and the elevator input (no stall effects)
        F_Lift = 0.5 * density * area * V_air_xz^2 * (c_L_0 + c_L_alpha * alpha + c_L_delta_e * delta_e)

        Drag force is modeled as a second order polynomial of the angle of attack
            (and therefore currently ignores the effect of the elevator input)
        F_Drag = 0.5 * density * area * V_air_xz^2 * (c_D_0 + c_D_alpha * alpha + c_D_alpha^2 * alpha^2)

        Elevator input is modeled as a linear function of the angle of attack to obtain a trim model
        delta_e = delta_0 + delta_alpha * alpha

        Coefficients for optimization:
        c_L_0, c_L_alpha, c_L_delta_e, c_D_0, c_D_alpha, c_D_alpha^2, delta_0, delta_alpha

        :param v_airspeed: airspeed in m/s
        :param angle_of_attack: angle of attack in rad

        :return: regression matrix X for the estimation of x- and z-forces for a single feature
        """

        # compute dynamic pressure times wing area
        const = 0.5 * self.air_density * self.area * \
            (v_airspeed[0]**2 + v_airspeed[2]**2)
        X_wing_aero_frame = np.zeros((3, 6))

        # Compute Drag force coeffiecients:
        X_wing_aero_frame[0, 3] = - const
        X_wing_aero_frame[0, 4] = - const * angle_of_attack
        X_wing_aero_frame[0, 5] = - const * (angle_of_attack ** 2)
        # Compute Lift force coefficients:
        X_wing_aero_frame[2, 0] = - const
        X_wing_aero_frame[2, 1] = - const * angle_of_attack
        X_wing_aero_frame[2, 2] = - const * elevator_input

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame
        X_wing_body_frame = X_wing_body_frame.transpose().flatten()
        return X_wing_body_frame

    def compute_wing_moment_features(self, v_airspeed, angle_of_attack, elevator_input, angular_velocity, angle_of_sideslip):
        """
        Model description:

        Compute pitching moment in stability axis frame.

        Pitching moment is modeled as a linear function of the angle of attack, the elevator input and the angle of sideslip
        M_Pitch = 0.5 * density * area * V_air_xz^2 * chord * (c_m_0 + c_m_alpha * alpha + c_m_delta_e * delta_e + c_mq * damping_feature)

        Coefficients for optimization:
        c_m_0, c_m_alpha, c_m_delta_e, c_mq

        :param v_airspeed: airspeed in m/s
        :param angle_of_attack: angle of attack in rad
        :param elevator_input: elevator input
        :param angular_velocity: angular velocity in rad/s
        :param angle_of_sideslip: angle of sideslip in rad

        :return: regression matrix X for the estimation of pitching moment for a single feature
        """

        vel_xz = np.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        const = 0.5 * self.air_density * self.area * \
            (vel_xz**2) * self.chord

        X_wing_aero_frame = np.zeros((3, 4))

        # Compute Pitching moment coefficients:
        X_wing_aero_frame[1, 0] = const
        X_wing_aero_frame[1, 1] = const * angle_of_attack
        X_wing_aero_frame[1, 2] = const * elevator_input
        X_wing_aero_frame[1, 3] = const * \
            (angular_velocity[1] * self.chord) / (2 * vel_xz)

        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame
        X_wing_body_frame = X_wing_body_frame.transpose().flatten()
        return X_wing_body_frame

    def compute_aero_force_features(self, v_airspeed_mat, angle_of_attack_vec, elevator_input_vec):
        """
        Inputs:
        :param v_airspeed_mat: airspeed in m/s with format numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        :param angle_of_attack_vec: angle of attack in rad with format vector of size (n) with corresponding AoA values

        Returns:
        :return: regression matrix X for the estimation of x- and z-forces
        """
        X_aero = self.compute_wing_force_features(
            v_airspeed_mat[0, :], angle_of_attack_vec[0], elevator_input_vec[0])
        aero_features_bar = Bar(
            'Feature Computation', max=v_airspeed_mat.shape[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_wing_force_features(
                v_airspeed_mat[i, :], angle_of_attack_vec[i], elevator_input_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
            aero_features_bar.next()
        aero_features_bar.finish()
        coef_dict = {
            "cl0": {"lin": {"x": "cl0_x", "y": "cl0_y", "z": "cl0_z"}},
            "clalpha": {"lin": {"x": "clalpha_x", "y": "clalpha_y", "z": "clalpha_z"}},
            "cldelta": {"lin": {"x": "cldelta_x", "y": "cldelta_y", "z": "cldelta_z"}},
            "cd0": {"lin": {"x": "cd0_x", "y": "cd0_y", "z": "cd0_z"}},
            "cdalpha": {"lin": {"x": "cdalpha_x", "y": "cdalpha_y", "z": "cdalpha_z"}},
            "cdalphasq": {"lin": {"x": "cdalphasq_x", "y": "cdalphasq_y", "z": "cdalphasq_z"}},
        }
        col_names = ["cl0_x", "cl0_y", "cl0_z",
                     "clalpha_x", "clalpha_y", "clalpha_z",
                     "cldelta_x", "cldelta_y", "cldelta_z",
                     "cd0_x", "cd0_y", "cd0_z",
                     "cdalpha_x", "cdalpha_y", "cdalpha_z",
                     "cdalphasq_x", "cdalphasq_y", "cdalphasq_z"]

        return X_aero, coef_dict, col_names

    def compute_aero_moment_features(self, v_airspeed_mat, angle_of_attack_vec, elevator_input_vec, angular_vel_mat, angle_of_sideslip_vec):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        """
        print("Starting computation of aero moment features...")
        X_aero = self.compute_wing_moment_features(
            v_airspeed_mat[0, :], angle_of_attack_vec[0], elevator_input_vec[0], angular_vel_mat[0, :], angle_of_sideslip_vec[0])
        aero_features_bar = Bar(
            'Feature Computatiuon', max=v_airspeed_mat.shape[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_wing_moment_features(
                v_airspeed_mat[i, :], angle_of_attack_vec[i], elevator_input_vec[i], angular_vel_mat[i, :], angle_of_sideslip_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
            aero_features_bar.next()
        aero_features_bar.finish()

        coef_dict = {
            "cm0": {"rot": {"x": "cm0_x", "y": "cm0_y", "z": "cm0_z"}},
            "cmalpha": {"rot": {"x": "cmalpha_x", "y": "cmalpha_y", "z": "cmalpha_z"}},
            "cmdelta": {"rot": {"x": "cmdelta_x", "y": "cmdelta_y", "z": "cmdelta_z"}},
            "cmq": {"rot": {"x": "cmq_x", "y": "cmq_y", "z": "cmq_z"}},
        }
        col_names = ["cm0_x", "cm0_y", "cm0_z",
                     "cmalpha_x", "cmalpha_y", "cmalpha_z",
                     "cmdelta_x", "cmdelta_y", "cmdelta_z",
                     "cmq_x", "cmq_y", "cmq_z",
                     ]
        return X_aero, coef_dict, col_names
