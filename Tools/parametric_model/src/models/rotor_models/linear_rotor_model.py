"""
 *
 * Copyright (c) 2023 Julius Schlapbach
 *               2023 Autonomous Systems Lab ETH Zurich
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

import numpy as np


class LinearRotorModel:
    def __init__(self, actuator_input_vec):
        """
        Inputs:
        actuator_input_vec: vector of actuator inputs (normalized between 0 and 1), numpy array of shape (n, 1)
        v_airspeed_mat: matrix of vertically stacked airspeed vectors, numpy array of shape (n, 3)
        rotor_axis_mat: matrices of rotor axis corresponding to different timestamps. only needed for models with tilting rotors.
        """
        self.throttle = actuator_input_vec

    def compute_actuator_force_matrix(self):
        print("Computing force features for rotor")
        coef_dict = {
            "ct": {"lin": {"x": "ct_x", "y": "ct_y", "z": "ct_z"}},
        }
        col_names = ["ct_x", "ct_y", "ct_z"]

        X_forces = np.zeros((self.throttle.shape[0], 3))
        X_forces[:, 0] = self.throttle

        return X_forces, coef_dict, col_names

    def compute_actuator_moment_matrix(self):
        print("Computing moment features for rotor")

        coef_dict = {
            "cmt": {"rot": {"x": "cmt_x", "y": "cmt_y", "z": "cmt_z"}},
        }
        col_names = ["cmt_x", "cmt_y", "cmt_z"]

        X_moments = np.zeros((self.throttle.shape[0], 3))
        X_moments[:, 1] = self.throttle

        return X_moments, coef_dict, col_names
