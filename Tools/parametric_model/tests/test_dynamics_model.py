__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from _pytest.python import Class
import pytest
from src.models import DynamicsModel
from src.tools.math_tools import rmse_between_numpy_arrays


def test_transformations():
    # Setup model with reference log
    req_topic_dict = {
        "vehicle_attitude": {"ulog_name": ["timestamp", "q[0]", "q[1]", "q[2]", "q[3]"],
                             "dataframe_name":  ["timestamp", "q0", "q1", "q2", "q3"]},
        "vehicle_local_position": {"ulog_name": ["timestamp", "ax", "ay", "az"]},
        "sensor_combined": {"ulog_name": ["timestamp", "accelerometer_m_s2[0]", "accelerometer_m_s2[1]", "accelerometer_m_s2[2]"]}}
    model = DynamicsModel(
        "resources/simple_quadrotor_model.ulg", req_topic_dict, 1)

    # Add gravity vector to inertial accelerations
    model.data_df["az"] -= 9.81

    # Transform inertial and body matrices to numpy matrices
    accel_NED_mat = model.data_df[["ax", "ay", "az"]].to_numpy()
    accel_FRD_mat = model.data_df[[
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
