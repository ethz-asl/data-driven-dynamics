__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.models.aerodynamic_models import TiltWingSection
from src.models.rotor_models import TiltingRotorModel
import os
import numpy as np
import math


def test_static_airspeed_computation_no_ang_vel():
    rotor_config_dict = {"description": "test rotor",
                         "dataframe_name": "u0",
                         "rotor_type": "TiltingRotorModel",
                         "tilt_actuator_dataframe_name": "u_tilt",
                         "rotor_axis": [1, 0, 0],
                         "tilt_axis": [0, 1, 0],
                         "max_tilt_angle_deg": 90,
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    tilt_actuator_vec = np.array([0, 0.5, 1, 0, 0.5, 1])
    actuator_input_vec = np.array([0.5, 0.5, 0.5, 1, 1, 1])
    v_airspeed_mat = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [
                              0, 0, -1], [0, 0, -1], [0, 0, -1]])
    rotor = TiltingRotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat,
                              tilt_actuator_vec)

    wing_section_config = {"rotor": "u3",
                           "control_surface_dataframe_name": "u8",
                           "wing_surface": 0.425,
                           "cp_position": [0, 0, 0],
                           "stall_angle_deg": 20,
                           "sig_scale_factor": 30,
                           "description": "wing test segment"}

    control_surface_output = np.array([0, 0, 0, 0, 0, 0])

    tilt_wing = TiltWingSection(
        wing_section_config, v_airspeed_mat, control_surface_output, rotor=rotor)

    assert (np.array_equal(tilt_wing.static_local_airspeed_mat, v_airspeed_mat))


def test_local_AoA():
    # thest the local angle of attack without rotor thrust
    rotor_config_dict = {"description": "test rotor",
                         "dataframe_name": "u0",
                         "rotor_type": "TiltingRotorModel",
                         "tilt_actuator_dataframe_name": "u_tilt",
                         "rotor_axis": [1, 0, 0],
                         "tilt_axis": [0, 1, 0],
                         "max_tilt_angle_deg": 90,
                         "turning_direction": 1,
                         "position": [0, 0, 0]}

    tilt_actuator_vec = np.array([0, 0.5, 1, 0, 0.5, 1])
    actuator_input_vec = np.array([0, 0, 0, 0, 0, 0])
    v_airspeed_mat = np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0], [
                              0, 0, -1], [0, 0, -1], [0, 0, -1]])
    rotor = TiltingRotorModel(rotor_config_dict, actuator_input_vec, v_airspeed_mat,
                              tilt_actuator_vec)
    X_moments, coef_list_moments = rotor.compute_actuator_force_matrix()

    wing_section_config = {"rotor": "u3",
                           "control_surface_dataframe_name": "u8",
                           "wing_surface": 0.425,
                           "cp_position": [0, 0, 0],
                           "stall_angle_deg": 20,
                           "sig_scale_factor": 30,
                           "description": "wing test segment"}

    control_surface_output = np.array([0, 0, 0, 0, 0, 0])

    tilt_wing = TiltWingSection(
        wing_section_config, v_airspeed_mat, control_surface_output, rotor=rotor)

    thrust_coef_list = [10, 0]
    # in degrees
    local_aoa_vec_des = np.array([0, 45, 90, -90, -45, 0])

    tilt_wing.update_local_airspeed_and_aoa(thrust_coef_list)
    assert (np.linalg.norm(tilt_wing.local_aoa_vec *
            180/math.pi - local_aoa_vec_des) < 10e-10)


if __name__ == "__main__":
    # set cwd to project directory when run as module
    cwd = os.getcwd()
    parent = os.path.join(cwd, os.pardir)
    des_cwd = os.path.join(parent, os.pardir)
    os.chdir(des_cwd)

    test_static_airspeed_computation_no_ang_vel()
    test_local_AoA()
