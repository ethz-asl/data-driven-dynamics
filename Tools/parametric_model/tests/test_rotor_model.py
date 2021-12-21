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

from src.models.rotor_models import RotorModel
import os
import numpy as np
import math


def test_local_airspeed_computation():
    rotor_config_dict = {"description": "test rotor",
                         "dataframe_name": "u0",
                         "rotor_type": "RotorModel",
                         "rotor_axis": [1, 0, 0],
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    actuator_input_vec = np.array([0, 0, 0])
    v_airspeed_mat = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]])

    rotor = RotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat)

    assert np.array_equal(rotor.v_air_parallel_abs, np.array([5, 0, 0]))
    assert np.array_equal(
        rotor.v_airspeed_perpendicular_to_rotor_axis[0, :], np.array([0, 0, 0]))
    assert np.array_equal(
        rotor.v_airspeed_perpendicular_to_rotor_axis[1, :], np.array([0, 5, 0]))
    assert np.array_equal(
        rotor.v_airspeed_perpendicular_to_rotor_axis[2, :], np.array([0, 0, 5]))


def test_rotor_thrust_prediction():
    rotor_config_dict = {"description": "test rotor",
                         "dataframe_name": "u0",
                         "rotor_type": "RotorModel",
                         "rotor_axis": [0, 0, -1],
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    actuator_input_vec = np.array([0, 0.5, 1, 0, 0.5, 1])
    v_airspeed_mat = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [
                              0, 0, -5], [0, 0, -5], [0, 0, -5]])
    thrust_coef_list = [-1, 10]

    correct_force_prediction = np.array([[0,       0,       0.],
                                         [0,       0,  -3.0625],
                                         [0,       0,   -12.25],
                                         [0,       0,       0.],
                                         [0,       0,       0.],
                                         [0,       0,  -6.125]])

    rotor = RotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat)
    X_forces, coef_dict, col_names = rotor.compute_actuator_force_matrix()
    print(rotor.predict_thrust_force(
        thrust_coef_list))
    assert (np.linalg.norm(rotor.predict_thrust_force(
        thrust_coef_list) - correct_force_prediction) < 10e-10)


if __name__ == "__main__":
    # set cwd to project directory when run as module
    cwd = os.getcwd()
    parent = os.path.join(cwd, os.pardir)
    des_cwd = os.path.join(parent, os.pardir)
    os.chdir(des_cwd)

    test_local_airspeed_computation()
    test_rotor_thrust_prediction()
