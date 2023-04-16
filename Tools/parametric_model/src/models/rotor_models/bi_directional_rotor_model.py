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

import numpy as np
import pandas as pd
import math
from progress.bar import Bar
from . import ChangingAxisRotorModel


class BiDirectionalRotorModel(ChangingAxisRotorModel):
    def __init__(
        self,
        rotor_config_dict,
        actuator_input_vec,
        v_airspeed_mat,
        air_density=1.225,
        angular_vel_mat=None,
    ):
        print(rotor_config_dict)
        self.n_timestamps = actuator_input_vec.shape[0]
        self.rotor_axis = np.array(rotor_config_dict["rotor_axis"]).reshape(3, 1)
        self.compute_rotor_axis_mat(actuator_input_vec)
        super(BiDirectionalRotorModel, self).__init__(
            rotor_config_dict,
            np.absolute(actuator_input_vec),
            v_airspeed_mat,
            air_density=1.225,
            angular_vel_mat=None,
        )

    def compute_rotor_axis_mat(self, actuator_input_vec):
        self.rotor_axis_mat = np.zeros((self.n_timestamps, 3))
        for i in range(self.n_timestamps):
            self.rotor_axis_mat[i, :] = self.rotor_axis * np.sign(actuator_input_vec[i])
