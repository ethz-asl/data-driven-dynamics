"""
 *
 * Copyright (c) 2021 Manuel Galliker
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

import numpy as np


class FuselageDragModel():
    def __init__(self, air_density=1.225):
        self.air_density = air_density

    def compute_single_fuselage_feature(self, v_airspeed):

        X_fuselage = np.zeros((3, 3))

        # Compute Drag force coeffiecients:
        X_fuselage[0, 0] = -0.5*self.air_density * \
            v_airspeed[0]*abs(v_airspeed[0])
        X_fuselage[1, 1] = -0.5*self.air_density * \
            v_airspeed[1]*abs(v_airspeed[1])
        X_fuselage[2, 2] = -0.5*self.air_density * \
            v_airspeed[2]*abs(v_airspeed[2])
        return X_fuselage

    def compute_fuselage_features(self, v_airspeed_mat):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        roll_commands = numpy array of dimension (n,3) with columns for [u_roll_1, u_roll_2, u_pitch_collective]
        """
        X_aero = self.compute_single_fuselage_feature(v_airspeed_mat[0, :])

        for i in range(1, v_airspeed_mat.shape[0]):
            X_curr = self.compute_single_fuselage_feature(v_airspeed_mat[i, :])
            X_aero = np.vstack((X_aero, X_curr))
        fuselage_coef_list = ["c_d_fuselage_x",
                              "c_d_fuselage_y", "c_d_fuselage_z"]
        return X_aero, fuselage_coef_list
