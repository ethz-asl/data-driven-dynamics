__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np

from src.tools.math_tools import cropped_sym_sigmoid
from scipy.spatial.transform import Rotation
from progress.bar import Bar

# AAE stands for aileron, aileron, elevator


class TiltWingSection():
    def __init__(self, wing_section_config, v_airspeed_mat, air_density=1.225, angular_vel_mat=None, rotor=None):
        self.stall_angle = wing_section_config["stall_angle_deg"]*math.pi/180.0
        self.sig_scale_fac = wing_section_config["sig_scale_factor"]
        self.air_density = air_density
        self.n_timestamps = v_airspeed_mat.shape[0]
        self.cp_position = np.array(
            wing_section_config["cp_position"]).reshape(3, 1)
        self.description = wing_section_config["description"]
        self.aero_coef_list = ["c_d_wing_xz_offset", "c_d_wing_xz_lin", "c_d_wing_xz_quad", "c_d_wing_xz_stall_min", "c_d_wing_xz_stall_90_deg",
                               "c_l_wing_xz_offset", "c_l_wing_xz_lin", "c_l_wing_xz_stall", "c_d_wing_y_offset"]

        # upstream rotor influencing the downwash over wing section.
        if rotor == None:
            self.in_downstream = False
        else:
            self.in_downstream = True
            self.rotor = rotor

        self.compute_static_local_airspeed(v_airspeed_mat, angular_vel_mat)

    def compute_static_local_airspeed(self, v_airspeed_mat, angular_vel_mat):
        """
            The only component of the airspeed chaning during optimization is the part due to the actuator downwash.
            The other "static" components (airspeed due to groundspeed, wind and angular velocity can be precomputed in this function"""

        if angular_vel_mat is not None:
            self.static_local_airspeed_mat = np.zeros(v_airspeed_mat.shape)
            assert (v_airspeed_mat.shape ==
                    angular_vel_mat.shape), "RotorModel: v_airspeed_mat and angular_vel_mat differ in size."
            for i in range(self.n_timestamps):
                self.static_local_airspeed_mat[i, :] = v_airspeed_mat[i, :] + \
                    np.cross(angular_vel_mat[i, :],
                             self.cp_position.flatten())

        else:
            self.static_local_airspeed_mat = v_airspeed_mat

    def update_local_airspeed_and_aoa(self, rotor_thrust_coef):

        self.local_airspeed_mat = np.zeros(
            self.static_local_airspeed_mat.shape)
        self.local_aoa_vec = np.zeros(self.static_local_airspeed_mat.shape[0])
        rotor_thrust_mat = self.rotor.predict_thrust_force(rotor_thrust_coef)
        if self.in_downstream:
            for i in range(self.n_timestamps):
                abs_thrust = np.linalg.norm(rotor_thrust_mat[i, :])
                v_air_parr = self.rotor.v_air_parallel_abs[i]
                v_downwash = self.rotor.rotor_axis * (-v_air_parr + math.sqrt(v_air_parr**2 + (
                    2*abs_thrust/(self.rotor.air_density*self.rotor.prop_area))))
                self.local_airspeed_mat[i, :] = self.static_local_airspeed_mat[i,
                                                                               :] + v_downwash.flatten()
                self.local_aoa_vec[i] = math.atan2(
                    self.local_airspeed_mat[i, 2],   self.local_airspeed_mat[i, 0])

    def predict_single_wing_segment_feature_mat(self, v_airspeed, angle_of_attack):
        """
        Model description:

        Compute lift and drag forces in stability axis frame.

        This is done by interpolating two models:
        1. More suffisticated Model for abs(AoA) < stall_angle

            - Lift force coefficient as linear function of AoA:
                F_Lift = 0.5 * density * area * \
                    V_air_xz^2 * (c_l_0 + c_l_lin*AoA)

            - Drag force coefficient as quadratic function of AoA
                F_Drag = 0.5 * density * area * V_air_xz^2 * \
                    (c_d_0 + c_d_lin * AoA + c_d_quad * AoA^2)

        2. Simple plate model for abs(AoA) > stall_angle
                F_Lift = density * area * V_air_xz^2 * \
                    cos(AoA) * sin(AoA) * c_l_stall
                F_Drag = 0.5 * density * area * \
                    V_air_xz^2 * sin(AoA) * c_d_stall

        The two models are interpolated with a symmetric sigmoid function obtained by multiplying two logistic functions:
            if abs(AoA) < stall_angle: cropped_sym_sigmoid(AoA) = 0
            if abs(AoA) > stall_angle: cropped_sym_sigmoid(AoA) = 1
        """

        v_xz = math.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        X_xz_aero_frame = np.zeros((3, 8))

        # Compute Drag force coeffiecients:
        X_xz_aero_frame[0, 0] = -(
            1 - cropped_sym_sigmoid(angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*v_xz**2
        X_xz_aero_frame[0, 1] = -(
            1 - cropped_sym_sigmoid(angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*angle_of_attack*v_xz**2
        X_xz_aero_frame[0, 2] = -(
            1 - cropped_sym_sigmoid(angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*angle_of_attack**2*v_xz**2
        X_xz_aero_frame[0, 3] = -(cropped_sym_sigmoid(angle_of_attack,
                                  x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*(1 - math.sin(angle_of_attack)**2)*v_xz**2
        X_xz_aero_frame[0, 3] = -(cropped_sym_sigmoid(angle_of_attack,
                                  x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*(1 - math.sin(angle_of_attack)**2)*v_xz**2
        X_xz_aero_frame[0, 4] = -(cropped_sym_sigmoid(angle_of_attack,
                                  x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*(math.sin(angle_of_attack)**2)*v_xz**2

        # Compute Lift force coefficients:
        X_xz_aero_frame[2, 5] = -(
            1 - cropped_sym_sigmoid(angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*angle_of_attack*v_xz**2
        X_xz_aero_frame[2, 6] = -(
            1 - cropped_sym_sigmoid(angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac))*v_xz**2
        X_xz_aero_frame[2, 7] = -cropped_sym_sigmoid(angle_of_attack, x_offset=self.stall_angle, scale_fac=self.sig_scale_fac) \
            * math.sin(2*angle_of_attack)*v_xz**2

        # Transorm from stability axis frame to body FRD frame
        R_aero_to_body = Rotation.from_rotvec(
            [0, -angle_of_attack, 0]).as_matrix()
        X_xz_body_frame = R_aero_to_body @ X_xz_aero_frame

        """
        Compute drag in y direction of body frame using a single coefficient: 
        F_y = 0.5 * density * area * V_air_y^2 * c_d_y"""
        X_y_body_frame = np.array([0, -math.copysign(
            1, v_airspeed[1]) * v_airspeed[1]**2, 0]).reshape(3, 1)
        X_wing_segment = np.hstack((X_xz_body_frame, X_y_body_frame))
        return X_wing_segment

    def predict_wing_segment_forces(self, rotor_thrust_coef, aero_force_coef):
        """
        Inputs:

        v_airspeed_mat: numpy array of dimension (n,3) with columns for [v_a_x, v_a_y, v_a_z]
        angle_of_attack_vec: vector of size (n) with corresponding AoA values
        flap_commands = numpy array of dimension (n,3) with columns for [u_aileron_right, u_aileron_left, u_elevator]
        """
        aero_force_coef = aero_force_coef.reshape(
            (aero_force_coef.shape[0], 1))
        self.update_local_airspeed_and_aoa(rotor_thrust_coef)
        X_aero = self.predict_single_wing_segment_feature_mat(
            self.local_airspeed_mat[0, :], self.local_aoa_vec[0])
        for i in range(1, self.n_timestamps):
            X_curr = self.predict_single_wing_segment_feature_mat(
                self.local_airspeed_mat[0, :], self.local_aoa_vec[0])
            X_aero = np.vstack((X_aero, X_curr))
        F_aero_force_pred = X_aero @ aero_force_coef
        return F_aero_force_pred
