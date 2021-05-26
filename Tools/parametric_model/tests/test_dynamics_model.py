__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

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
    data_handler.loadLog("resources/quadrotor_model.ulg")

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

