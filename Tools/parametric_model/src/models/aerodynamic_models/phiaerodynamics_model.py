"""
 *
 * Copyright (c) 2023 Jaeyoung Lim
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

__author__ = "Jaeyoung Lim"
__maintainer__ = "Jaeyoung Lim"
__license__ = "BSD 3"

import math
import numpy as np

from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation
from progress.bar import Bar

"""
The PhiAerodynamics model is a global singularity free aerodynamics model

[1] Lustosa, Leandro R., François Defaÿ, and Jean-Marc Moschetta. "Global singularity-free aerodynamic model for
 algorithmic flight control of tail sitters." Journal of Guidance, Control, and Dynamics 42.2 (2019): 303-316.
 """

class PhiAerodynamicsModel():
    def __init__(self, config_dict):
        self.stall_angle = config_dict["stall_angle_deg"]*math.pi/180.0
        self.air_density = 1.225
        self.area = config_dict["area"]
        self.reference_wingspan = 2.0
        self.reference_wingchord = 0.5

    def compute_wing_force_features(self, v_airspeed, angle_of_attack):
        X_wing_body_frame = np.zeros((3, 9))
        eta = np.sqrt(v_airspeed[0]**2 + v_airspeed[1]**2 + v_airspeed[2]**2)  # Did not take into account the angular velocity
        constant = - 0.5 * self.air_density * self.area * eta
        X_wing_body_frame[0, 0] = constant * v_airspeed[0]
        X_wing_body_frame[0, 1] = constant * v_airspeed[1]
        X_wing_body_frame[0, 2] = constant * v_airspeed[2]
        X_wing_body_frame[1, 3] = constant * v_airspeed[0]
        X_wing_body_frame[1, 4] = constant * v_airspeed[1]
        X_wing_body_frame[1, 5] = constant * v_airspeed[2]
        X_wing_body_frame[2, 6] = constant * v_airspeed[0]
        X_wing_body_frame[2, 7] = constant * v_airspeed[1]
        X_wing_body_frame[2, 8] = constant * v_airspeed[2]

        X_wing_body_frame = X_wing_body_frame.flatten('F')
        return X_wing_body_frame

    def compute_wing_moment_features(self, v_airspeed, angle_of_attack, angle_of_sideslip, elevator_input):
        X_wing_body_frame = np.zeros((3, 10))
        eta = np.sqrt(v_airspeed[0]**2 + v_airspeed[1]**2 + v_airspeed[2]**2)  # TODO Take dynamic pressure
        constant = - 0.5 * self.air_density * self.area * eta
        X_wing_body_frame[0, 0] = self.reference_wingspan**2 * constant * v_airspeed[0]
        X_wing_body_frame[0, 1] = self.reference_wingspan**2 * constant * v_airspeed[1]
        X_wing_body_frame[0, 2] = self.reference_wingspan**2 * constant * v_airspeed[2]
        X_wing_body_frame[1, 3] = self.reference_wingchord**2 * constant * v_airspeed[0]
        X_wing_body_frame[1, 4] = self.reference_wingchord**2 * constant * v_airspeed[1]
        X_wing_body_frame[1, 5] = self.reference_wingchord**2 * constant * v_airspeed[2]
        X_wing_body_frame[2, 6] = self.reference_wingspan**2 * constant * v_airspeed[0]
        X_wing_body_frame[2, 7] = self.reference_wingspan**2 * constant * v_airspeed[1]
        X_wing_body_frame[2, 8] = self.reference_wingspan**2 * constant * v_airspeed[2]
        X_wing_body_frame[1, 9] = elevator_input
        X_wing_body_frame = X_wing_body_frame.flatten('F')
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
            "phifv_11": {"lin":{ "x": "phifv_11_x","y": "phifv_11_y","z":"phifv_11_z"}},
            "phifv_12": {"lin":{ "x": "phifv_12_x","y": "phifv_12_y","z":"phifv_12_z"}},
            "phifv_13": {"lin":{ "x": "phifv_13_x","y": "phifv_13_y","z":"phifv_13_z"}},
            "phifv_21": {"lin":{ "x": "phifv_21_x","y": "phifv_21_y","z":"phifv_21_z"}},
            "phifv_22": {"lin":{ "x": "phifv_22_x","y": "phifv_22_y","z":"phifv_22_z"}},
            "phifv_23": {"lin":{ "x": "phifv_23_x","y": "phifv_23_y","z":"phifv_23_z"}},
            "phifv_31": {"lin":{ "x": "phifv_31_x","y": "phifv_31_y","z":"phifv_31_z"}},
            "phifv_32": {"lin":{ "x": "phifv_32_x","y": "phifv_32_y","z":"phifv_32_z"}},
            "phifv_33": {"lin":{ "x": "phifv_33_x","y": "phifv_33_y","z":"phifv_33_z"}},
        }
        col_names = ["phifv_11_x", "phifv_11_y", "phifv_11_z", 
                    "phifv_12_x", "phifv_12_y", "phifv_12_z",
                    "phifv_13_x", "phifv_13_y", "phifv_13_z",
                    "phifv_21_x", "phifv_21_y", "phifv_21_z",
                    "phifv_22_x", "phifv_22_y", "phifv_22_z",
                    "phifv_23_x", "phifv_23_y", "phifv_23_z",
                    "phifv_31_x", "phifv_31_y", "phifv_31_z", 
                    "phifv_32_x", "phifv_32_y", "phifv_32_z",
                    "phifv_33_x", "phifv_33_y", "phifv_33_z"]
                
        return X_aero, coef_dict, col_names

    def compute_aero_moment_features(self, v_airspeed_mat, angle_of_attack_vec, angle_of_sideslip_vec, elevator_input_vec):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        """
        print("Starting computation of aero moment features...")
        X_aero = self.compute_wing_moment_features(
            v_airspeed_mat[0, :], angle_of_attack_vec[0], angle_of_sideslip_vec[0], elevator_input_vec[0])
        aero_features_bar = Bar(
            'Feature Computatiuon', max=v_airspeed_mat.shape[0])
        for i in range(1, len(angle_of_attack_vec)):
            X_curr = self.compute_wing_moment_features(
                v_airspeed_mat[i, :], angle_of_attack_vec[i], angle_of_sideslip_vec[i], elevator_input_vec[i])
            X_aero = np.vstack((X_aero, X_curr))
            aero_features_bar.next()
        aero_features_bar.finish()
        coef_dict = {
            "phimv_11": {"rot":{ "x": "phimv_11_x","y": "phimv_11_y","z":"phimv_11_z"}},
            "phimv_12": {"rot":{ "x": "phimv_12_x","y": "phimv_12_y","z":"phimv_12_z"}},
            "phimv_13": {"rot":{ "x": "phimv_13_x","y": "phimv_13_y","z":"phimv_13_z"}},
            "phimv_21": {"rot":{ "x": "phimv_21_x","y": "phimv_21_y","z":"phimv_21_z"}},
            "phimv_22": {"rot":{ "x": "phimv_22_x","y": "phimv_22_y","z":"phimv_22_z"}},
            "phimv_23": {"rot":{ "x": "phimv_23_x","y": "phimv_23_y","z":"phimv_23_z"}},
            "phimv_31": {"rot":{ "x": "phimv_31_x","y": "phimv_31_y","z":"phimv_31_z"}},
            "phimv_32": {"rot":{ "x": "phimv_32_x","y": "phimv_32_y","z":"phimv_32_z"}},
            "phimv_33": {"rot":{ "x": "phimv_33_x","y": "phimv_33_y","z":"phimv_33_z"}},
            "cmdelta": {"rot":{ "x": "cmdelta_x","y": "cmdelta_y","z":"cmdelta_z"}},
        }
        col_names = ["phimv_11_x", "phimv_11_y", "phimv_11_z", 
                    "phimv_12_x", "phimv_12_y", "phimv_12_z",
                    "phimv_13_x", "phimv_13_y", "phimv_13_z",
                    "phimv_21_x", "phimv_21_y", "phimv_21_z",
                    "phimv_22_x", "phimv_22_y", "phimv_22_z",
                    "phimv_23_x", "phimv_23_y", "phimv_23_z",
                    "phimv_31_x", "phimv_31_y", "phimv_31_z", 
                    "phimv_32_x", "phimv_32_y", "phimv_32_z",
                    "phimv_33_x", "phimv_33_y", "phimv_33_z",
                    "cmdelta_x", "cmdelta_y", "cmdelta_z"]
        return X_aero, coef_dict, col_names
