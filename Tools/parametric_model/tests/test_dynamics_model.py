__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.tools.math_tools import rmse_between_numpy_arrays
from src.models import ModelConfig
from src.models import DynamicsModel
import os


def test_transformations():
    config = ModelConfig("dynamics_model_test_config.yaml")
    # Setup model with reference log
    model = DynamicsModel(config_dict=config.dynamics_model_config,
                          rel_data_path="resources/quadrotor_model.ulg")
    # Add gravity vector to inertial accelerations
    model.data_df["az"] -= 9.81

    # Transform inertial and body matrices to numpy matrices
    accel_NED_mat = model.data_df[["ax", "ay", "az"]].to_numpy()
    accel_FRD_mat = model.data_df[[
        "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]].to_numpy()

    # Check transform from inertial NED to FRD body frame:
    accel_NED_transformed_to_FRD = model.rot_to_body_frame(
        accel_NED_mat)
    assert (rmse_between_numpy_arrays(
        accel_FRD_mat, accel_NED_transformed_to_FRD) <= 0.1)

    # Check transform from FRD body frame to inertial NED frame:
    accel_FRD_transformed_to_NED = model.rot_to_world_frame(
        accel_FRD_mat)
    assert (rmse_between_numpy_arrays(
        accel_NED_mat, accel_FRD_transformed_to_NED) <= 0.1)

    return


# Run uas module using 'python3 -m tests/test_dynamics_model' for development and testing of the test.
if __name__ == "__main__":
    # set cwd to project directory when run as module
    cwd = os.getcwd()
    parent = os.path.join(cwd, os.pardir)
    des_cwd = os.path.join(parent, os.pardir)
    os.chdir(des_cwd)

    test_transformations()
