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
    def __init__(self, config_dict, mass):
        self.air_density = 1.225
        self.ref_area = config_dict["area"]
        self.mass = mass
        self.gravity = 9.81

    def compute_wing_force_features(self, v_airspeed, angle_of_attack, elevator_input):
        """
        Model description:

        Compute lift and drag forces in stability axis frame.

        Lift force is modeled as a linear function of the angle of attack and the elevator input (no stall effects)
        F_Lift = 0.5 * density * area * V_air_xz^2 * (c_L_0 + c_L_alpha * alpha + c_L_delta_e * delta_e)

        Drag force is modeled as a quadratic function of the lift coefficient
        F_Drag = 0.5 * density * area * V_air_xz^2 * (c_D_0 + 1 / (pi * e * ar) * c_L^2)

        Coefficients for optimization:
        c_L_0, c_L_alpha, c_L_delta_e, c_D_0, 1/ar

        :param v_airspeed: airspeed in m/s
        :param angle_of_attack: angle of attack in rad

        :return: regression matrix X for the estimation of x- and z-forces for a single feature
        """

        # compute dynamic pressure times wing area
        dyn_pressure = 0.5 * self.air_density * (v_airspeed[0]**2 + v_airspeed[2]**2)
        X_wing_aero_frame = np.zeros((3, 5))

        # TODO
        # # features for drag coefficient computation
        # cL = 2 * self.mass * (self.gravity * np.cos(flight_path_angle) -
        #                       angular_acceleration[2]) / (self.air_density * self.ref_area * (v_airspeed[0]**2 + v_airspeed[2]**2))
        # X_wing_aero_frame[0, 3] = - dyn_pressure
        # X_wing_aero_frame[0, 4] = - (1 / (np.pi * np.e)) * dyn_pressure * cL**2

        # features for lift coefficient computation
        X_wing_aero_frame[2, 0] = - dyn_pressure * self.ref_area
        X_wing_aero_frame[2, 1] = - dyn_pressure * self.ref_area * angle_of_attack
        X_wing_aero_frame[2, 2] = - dyn_pressure * self.ref_area * elevator_input

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, angle_of_attack, 0]).as_matrix()
        X_wing_body_frame = R_aero_to_body @ X_wing_aero_frame
        X_wing_body_frame = X_wing_body_frame.flatten()
        return X_wing_body_frame

    # def compute_wing_moment_features(self, v_airspeed, angle_of_attack, angle_of_sideslip):
    # TODO: implement

    def compute_aero_force_features(self, v_airspeed_mat, angle_of_attack_vec, elevator_vec, gamma_vec, ang_vel_mat):
        """
        Inputs:
        :param v_airspeed_mat: airspeed in m/s with format numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        :param angle_of_attack_vec: angle of attack in rad with format vector of size (n) with corresponding AoA values

        Returns:
        :return: regression matrix X for the estimation of x- and z-forces
        """
        X_aero = self.compute_wing_force_features(
            v_airspeed_mat[0, :], angle_of_attack_vec[0], elevator_vec[0])
        aero_features_bar = Bar(
            'Feature Computation', max=v_airspeed_mat.shape[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_wing_force_features(
                v_airspeed_mat[i, :], angle_of_attack_vec[i], elevator_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
            aero_features_bar.next()
        aero_features_bar.finish()

        coef_dict = {
            'c_L_0': {"lin": {"x": 'c_L_0_x', "y": 'c_L_0_y', "z": 'c_L_0_z'}},
            'c_L_alpha': {"lin": {"x": 'c_L_alpha_x', "y": 'c_L_alpha_y', "z": 'c_L_alpha_z'}},
            'c_L_delta': {"lin": {"x": 'c_L_delta_x', "y": 'c_L_delta_y', "z": 'c_L_delta_z'}},
            'c_D_0': {"lin": {"x": 'c_D_0_x', "y": 'c_D_0_y', "z": 'c_D_0_z'}},
            'inv_ar': {"lin": {"x": 'inv_ar_x', "y": 'inv_ar_y', "z": 'inv_ar_z'}},
        }
        col_names = ["c_L_0_x", "c_L_0_y", "c_L_0_z",
                     "c_L_alpha_x", "c_L_alpha_y", "c_L_alpha_z",
                     "c_L_delta_x", "c_L_delta_y", "c_L_delta_z",
                     "c_D_0_x", "c_D_0_y", "c_D_0_z",
                     "inv_ar_x", "inv_ar_y", "inv_ar_z"]


        return X_aero, coef_dict, col_names

    # def compute_aero_moment_features(self, v_airspeed_mat, angle_of_attack_vec, angle_of_sideslip_vec):
    # TODO: implement
