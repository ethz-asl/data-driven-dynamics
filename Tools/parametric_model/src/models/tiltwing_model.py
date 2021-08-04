__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

"""Model to estimate the system parameters of gazebos standart vtol quadplane:
https://docs.px4.io/master/en/simulation/gazebo_vehicles.html#standard_vtol """


import numpy as np
import pandas as pd
import math

from .dynamics_model import DynamicsModel
from scipy.optimize import minimize
from scipy.linalg import block_diag
from .model_plots import model_plots, quad_plane_model_plots
from .model_config import ModelConfig
from .aerodynamic_models import TiltWingSection, FuselageDragModel, ElevatorModel
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


"""This model estimates forces and moments for quad plane as for example the standard vtol in gazebo."""


class TiltWingModel(DynamicsModel):
    def __init__(self, config_file="tilt_wing_config.yaml"):
        self.config = ModelConfig(config_file)
        super(TiltWingModel, self).__init__(
            config_dict=self.config.dynamics_model_config)

        self.model_name = "tilt_wing_model"

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.stall_angle = math.pi/180 * \
            self.config.model_config["aerodynamics"]["stall_angle_deg"]
        self.sig_scale_fac = self.config.model_config["aerodynamics"]["sig_scale_factor"]
        self.mass = self.config.model_config["mass"]

        self.visual_dataframe_selector_config_dict = {
            "x_axis_col": "timestamp",
            "sub_plt1_data": ["q0", "q1", "q2", "q3"],
            "sub_plt2_data": ["u0", "u1", "u2", "u3", "u4"],
            "sub_plt3_data": ["u5", "u6", "u7", "u8", "u9", "u_tilt"]}

    def load_dataframes(self, data_frames):
        self.data_df = data_frames
        self.u_tilt_vec = self.data_df["u_tilt"].to_numpy()

        self.n_samples = self.data_df.shape[0]
        self.quaternion_df = self.data_df[["q0", "q1", "q2", "q3"]]
        self.q_mat = self.quaternion_df.to_numpy()

        if "V_air_body_x" not in self.data_df:
            print("computing airspeed")
            self.normalize_actuators(
                ["actuator_controls_0", "actuator_controls_1"], control_outputs_used=True)
            self.compute_airspeed_from_groundspeed(["vx", "vy", "vz"])

        assert (self.estimate_moments ==
                False), "Estimation of moments is not yet implemented in TiltWingModel. Disable in config file to estimate forces."

    def prepare_regression_matrices(self):

        airspeed_mat = self.data_df[["V_air_body_x",
                                     "V_air_body_y", "V_air_body_z"]].to_numpy()
        # Rotor features as self.X_rotor_forces
        self.compute_rotor_features(self.rotor_config_dict)

        # Initialize fuselage drag model
        self.fuselage_model = FuselageDragModel()
        self.X_fuselage, self.fuselage_coef_list = self.fuselage_model.compute_fuselage_features(
            airspeed_mat)

        # Initialize elevator model
        self.elevator_model = ElevatorModel()
        self.X_elevator, self.elevator_coef_list = self.elevator_model.compute_elevator_features(
            airspeed_mat, self.data_df["u9"].to_numpy(), self.data_df["AoA"].to_numpy())
        # Initialize wing sections
        self.wing_sections = []
        wing_sections_config_list = self.config.model_config["aerodynamics"]["main_wing_sections"]
        for i in range(len(wing_sections_config_list)):
            curr_wing_section_config = wing_sections_config_list[i]
            curr_rotor = self.rotor_dict["wing_"][(
                curr_wing_section_config["rotor"])]
            curr_control_surface_output = self.data_df[curr_wing_section_config["control_surface_dataframe_name"]]
            curr_wing_section = TiltWingSection(
                curr_wing_section_config, airspeed_mat, curr_control_surface_output, rotor=curr_rotor)
            self.wing_sections.append(curr_wing_section)
            self.aero_coef_list = curr_wing_section.aero_coef_list

        # Accelerations
        accel_body_mat = self.data_df[[
            "acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
        self.y_accel = accel_body_mat.flatten()
        self.y_forces = self.mass * accel_body_mat.flatten()

    def predict_forces(self, x):

        fuselage_coef = np.array(x[0:3]).reshape(3, 1)
        wing_coef = np.array(x[3:13]).reshape(10, 1)
        elevator_coef = np.array(x[13]).reshape(1, 1)
        main_wing_rotor_thrust_coef = np.array(x[[18, 19]]).reshape(2, 1)
        rotor_coef = np.array(x[14:20]).reshape(6, 1)
        F_aero_pred = np.zeros((self.y_forces.shape[0], 1))
        for wing_section in self.wing_sections:
            F_aero_segment_pred = wing_section.predict_wing_segment_forces(
                main_wing_rotor_thrust_coef, wing_coef)
            F_aero_pred = np.add(F_aero_pred, F_aero_segment_pred)

        F_rotor_pred = self.X_rotor_forces @ rotor_coef
        F_fuselage = self.X_fuselage @ fuselage_coef
        F_elevator = self.X_elevator @ elevator_coef
        F_pred = np.add(F_aero_pred, F_rotor_pred)
        F_pred = np.add(F_pred, F_fuselage)
        F_pred = np.add(F_pred, F_elevator)
        return F_pred

    def objective(self, x):
        print("Evaluating objective function.")
        F_pred = self.predict_forces(x)
        a_pred = F_pred.flatten() / self.mass
        # pred_error = np.linalg.norm((a_pred - self.y_accel))
        pred_error = np.sqrt(((a_pred - self.y_accel) ** 2).mean())
        print("acceleration prediction RMSE: ", pred_error)
        return pred_error

    def estimate_model(self):
        print("Estimating quad plane model using the following data:")
        print(self.data_df.columns)
        self.data_df_len = self.data_df.shape[0]
        print("resampled data contains ", self.data_df_len, "timestamps.")
        self.prepare_regression_matrices()

        # optimization_variables:
        optimization_parameters = self.config.model_config["optimzation_parameters"]
        self.coef_list = self.fuselage_coef_list + self.aero_coef_list + \
            self.elevator_coef_list + self.rotor_forces_coef_list
        config_coef_list = (
            optimization_parameters["initial_coefficients"]).keys()
        print("estimating coefficients: ", self.coef_list)
        assert (self.coef_list == list(config_coef_list)), (
            "keys on config file differ from coefficient list: ",  self.coef_list, config_coef_list)

        # initial guesses
        x0 = np.array(
            list(optimization_parameters["initial_coefficients"].values()))
        self.coef_name_list = list(
            optimization_parameters["initial_coefficients"].keys())
        print("Initial coefficients: ", x0)
        coef_bounds = list(
            optimization_parameters["coefficient_bounds"].values())
        print("Coefficients bounds: ", coef_bounds)

        print(self.objective(x0))

        solution = minimize(self.objective, x0, method='TNC',
                            bounds=coef_bounds)
        self.x_opt = solution.x

        metrics_dict = {"R2": float(r2_score(self.y_forces, self.predict_forces(self.x_opt))),
                        "RMSE": float(self.objective(self.x_opt))}
        model_dict = {"config": self.config}

        self.generate_model_dict(
            self.x_opt, metrics_dict, model_dict)
        self.save_result_dict_to_yaml(file_name="tiltwing_model")
        return

    def plot_model_predicitons(self):

        wing_local_airspeed = self.wing_sections[0].local_airspeed_mat

        model_plots.plot_airspeed_and_AoA(
            np.hstack((self.wing_sections[0].static_local_airspeed_mat, self.wing_sections[0].local_aoa_vec.reshape(wing_local_airspeed.shape[0], 1))), self.data_df["timestamp"])
        model_plots.plot_airspeed_and_AoA(
            np.hstack((self.wing_sections[0].local_airspeed_mat, self.wing_sections[0].local_aoa_vec.reshape(wing_local_airspeed.shape[0], 1))), self.data_df["timestamp"])
        model_plots.plot_airspeed_and_AoA(
            self.data_df[["V_air_body_x", "V_air_body_y", "V_air_body_z", "AoA"]], self.data_df["timestamp"])

        y_accel_pred = self.predict_forces(self.x_opt)/self.mass
        model_plots.plot_accel_predeictions(
            self.y_accel, y_accel_pred, self.data_df["timestamp"])
        model_plots.plot_accel_and_airspeed_in_z_direction(
            self.y_accel, y_accel_pred, self.data_df["V_air_body_z"], self.data_df["timestamp"])

        model_plots.plot_accel_and_airspeed_in_y_direction(
            self.y_accel, y_accel_pred, self.data_df["V_air_body_y"], self.data_df["timestamp"])

        plt.show()
        return
