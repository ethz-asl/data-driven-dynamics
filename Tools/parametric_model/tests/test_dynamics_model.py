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

from _pytest.python import Class
import pytest
from src.models import DynamicsModel
from src.models import ModelConfig
from src.tools import DataHandler
from src.tools.math_tools import rmse_between_numpy_arrays
import os
from pathlib import Path


def test_transformations(config_file="dynamics_model_test_config.yaml"):
    rel_config_file_path = "Tools/parametric_model/configs/" + config_file
    proj_path = Path(os.getcwd())
    config_file_path = os.path.join(proj_path, rel_config_file_path)
    config = ModelConfig(config_file_path)

    data_handler = DataHandler(config_file_path)
    data_handler.loadLogs("resources/quadrotor_model.ulg")

    data_df = data_handler.get_dataframes()

    # Setup model with reference log
    model = DynamicsModel(config_dict=config.dynamics_model_config)
    model.load_dataframes(data_df)

    # Add gravity vector to inertial accelerations
    data_df["az"] -= 9.81

    # Transform inertial and body matrices to numpy matrices
    accel_NED_mat = data_df[["ax", "ay", "az"]].to_numpy()
    accel_FRD_mat = data_df[[
        "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]].to_numpy()

    # Check transform from inertial NED to FRD body frame:
    accel_NED_transformed_to_FRD = model.rot_to_body_frame(accel_NED_mat)
    assert (rmse_between_numpy_arrays(
        accel_FRD_mat, accel_NED_transformed_to_FRD) <= 0.1)

    # Check transform from FRD body frame to inertial NED frame:
    accel_FRD_transformed_to_NED = model.rot_to_world_frame(accel_FRD_mat)
    assert (rmse_between_numpy_arrays(
        accel_NED_mat, accel_FRD_transformed_to_NED) <= 0.1)

    return
