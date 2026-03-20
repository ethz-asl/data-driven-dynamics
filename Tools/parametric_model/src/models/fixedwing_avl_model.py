"""
 *
 * Copyright (c) 2024 Jaeyoung Lim
 *               2024 Autonomous Systems Lab ETH Zurich
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

__author__ = "Jaeyoung Lim"
__maintainer__ = "Jaeyoung Lim"
__license__ = "BSD 3"


import numpy as np

from . import aerodynamic_models
from .dynamics_model import DynamicsModel
from .model_config import ModelConfig
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation


class FixedWingAVLModel(DynamicsModel):
    """
    Fixed-wing aircraft model using AVL (Athena Vortex Lattice) aerodynamics.
    
    This model extends the basic fixed-wing model to support the comprehensive
    AVL aerodynamic model with:
    - Multiple control surfaces (not just elevator)
    - Body rate effects on forces (not just moments)
    - Sideslip effects on forces (not just moments)
    - Full 6-DOF aerodynamic derivatives
    """
    
    def __init__(
        self, config_file, normalization=True, model_name="avl_fixedwing_model"
    ):
        self.config = ModelConfig(config_file)
        super(FixedWingAVLModel, self).__init__(
            config_dict=self.config.dynamics_model_config, normalization=normalization
        )
        self.mass = self.config.model_config["mass"]
        self.moment_of_inertia = np.diag(
            [
                self.config.model_config["moment_of_inertia"]["Ixx"],
                self.config.model_config["moment_of_inertia"]["Iyy"],
                self.config.model_config["moment_of_inertia"]["Izz"],
            ]
        )

        self.model_name = model_name

        self.rotor_config_dict = self.config.model_config["actuators"]["rotors"]
        self.aerodynamics_dict = self.config.model_config["aerodynamics"]

        # Initialize AVL aerodynamic model
        try:
            self.aero_model = getattr(
                aerodynamic_models, self.aerodynamics_dict["type"]
            )(self.aerodynamics_dict)
        except AttributeError:
            error_str = (
                "Aerodynamics Model '{0}' not found, is it added to models "
                "directory and models/__init__.py?".format(self.aerodynamics_dict["type"])
            )
            raise AttributeError(error_str)
        
        # Verify it's actually an AVL model
        if self.aerodynamics_dict["type"] != "AVLModel":
            print(f"Warning: FixedWingAVLModel expected AVLModel, got {self.aerodynamics_dict['type']}")
        
        # Get control surface configuration
        self.num_ctrl_surfaces = self.aerodynamics_dict.get("num_ctrl_surfaces", 0)
        self.ctrl_surface_names = self.aerodynamics_dict.get("ctrl_surface_names", [])
        
        print(f"Initialized AVL model with {self.num_ctrl_surfaces} control surfaces: {self.ctrl_surface_names}")

    def prepare_control_deflections(self):
        """
        Prepare control surface deflection matrix from dataframe.
        
        Returns control deflections for all configured control surfaces.
        """
        n_samples = len(self.data_df)
        control_deflections = np.zeros((n_samples, self.num_ctrl_surfaces))
        
        # Map control surface names to dataframe columns
        for i, ctrl_name in enumerate(self.ctrl_surface_names):
            if ctrl_name.lower() in self.data_df.columns:
                control_deflections[:, i] = self.data_df[ctrl_name.lower()].to_numpy()
            else:
                print(f"Warning: Control surface '{ctrl_name}' not found in dataframe. Available: {list(self.data_df.columns)}")
                print(f"Using zero deflection for {ctrl_name}")
        
        return control_deflections

    def prepare_force_regression_matrices(self):
        """
        Prepare regression matrices for force estimation using AVL model.
        
        The AVL model requires:
        - Airspeed (3D vector)
        - Angle of attack
        - Angle of sideslip (needed for forces in AVL model)
        - Angular velocities (needed for forces in AVL model)
        - Control surface deflections (all surfaces)
        """
        accel_mat = self.data_df[["acc_b_x", "acc_b_y", "acc_b_z"]].to_numpy()
        force_mat = accel_mat * self.mass
        self.y_forces = (force_mat).flatten()
        self.data_df[
            ["measured_force_x", "measured_force_y", "measured_force_z"]
        ] = force_mat

        # Prepare AVL model inputs
        airspeed_mat = self.data_df[
            ["V_air_body_x", "V_air_body_y", "V_air_body_z"]
        ].to_numpy()
        aoa_vec = self.data_df["angle_of_attack"].to_numpy()
        sideslip_vec = self.data_df["angle_of_sideslip"].to_numpy()
        angular_vel_mat = self.data_df[
            ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
        ].to_numpy()
        
        # Get control deflections for all surfaces
        control_deflections_mat = self.prepare_control_deflections()

        # Compute AVL aerodynamic force features
        (
            X_aero,
            coef_dict_aero,
            col_names_aero,
        ) = self.aero_model.compute_aero_force_features(
            airspeed_mat,
            aoa_vec,
            sideslip_vec,
            angular_vel_mat,
            control_deflections_mat
        )
        
        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)
        self.y_dict.update(
            {
                "lin": {
                    "x": "measured_force_x",
                    "y": "measured_force_y",
                    "z": "measured_force_z",
                }
            }
        )

    def prepare_moment_regression_matrices(self):
        """
        Prepare regression matrices for moment estimation using AVL model.
        
        The AVL model requires:
        - Airspeed (3D vector)
        - Angle of attack
        - Angle of sideslip
        - Angular velocities
        - Control surface deflections (all surfaces)
        """
        # Angular acceleration
        moment_mat = np.matmul(
            self.data_df[["ang_acc_b_x", "ang_acc_b_y", "ang_acc_b_z"]].to_numpy(),
            self.moment_of_inertia,
        )
        self.y_moments = moment_mat.flatten()
        self.data_df[
            ["measured_moment_x", "measured_moment_y", "measured_moment_z"]
        ] = moment_mat

        # Prepare AVL model inputs
        airspeed_mat = self.data_df[
            ["V_air_body_x", "V_air_body_y", "V_air_body_z"]
        ].to_numpy()
        aoa_vec = self.data_df["angle_of_attack"].to_numpy()
        sideslip_vec = self.data_df["angle_of_sideslip"].to_numpy()
        angular_vel_mat = self.data_df[
            ["ang_vel_x", "ang_vel_y", "ang_vel_z"]
        ].to_numpy()
        
        # Get control deflections for all surfaces
        control_deflections_mat = self.prepare_control_deflections()

        # Compute AVL aerodynamic moment features
        (
            X_aero,
            coef_dict_aero,
            col_names_aero,
        ) = self.aero_model.compute_aero_moment_features(
            airspeed_mat,
            aoa_vec,
            sideslip_vec,
            angular_vel_mat,
            control_deflections_mat
        )

        self.data_df[col_names_aero] = X_aero
        self.coef_dict.update(coef_dict_aero)

        self.y_dict.update(
            {
                "rot": {
                    "x": "measured_moment_x",
                    "y": "measured_moment_y",
                    "z": "measured_moment_z",
                }
            }
        )
    
    def sanity_check_parameters(self, identified_params):
        """
        Perform sanity checks comparing estimated parameters with theoretical values.
        
        Compares:
        1. CD_induced vs theoretical induced drag from CLa
        2. Post-stall parameters vs theoretical nonlinear functions
        3. Overall parameter reasonableness
        """
        print("\n" + "="*79)
        print("                    AVL Model Sanity Checks")
        print("="*79)
        
        # Extract relevant parameters
        params = identified_params
        
        # Get aerodynamic parameters
        CLa = params.get('CLa', 0)
        CL0 = params.get('CL0', 0)
        CD0 = params.get('CD0', 0)
        CD_induced = params.get('CD_induced', 0)
        CD_poststall = params.get('CD_poststall', 0)
        CL_poststall = params.get('CL_poststall', 0)
        
        # Get geometry from config
        AR = self.aerodynamics_dict.get('AR', 10.0)
        eff = self.aerodynamics_dict.get('eff', 0.95)
        alpha_stall = np.rad2deg(self.aerodynamics_dict.get('alpha_stall', np.deg2rad(15)))
        
        print("\n1. Induced Drag Check")
        print("-" * 79)
        print(f"   Oswald efficiency (e): {eff:.3f}")
        print(f"   Aspect ratio (AR): {AR:.2f}")
        print(f"   Identified CLa: {CLa:.4f} /rad")
        
        # Theoretical induced drag coefficient at typical cruise alpha (5 deg)
        alpha_cruise = np.deg2rad(5.0)
        CL_cruise_theory = CL0 + CLa * alpha_cruise
        CD_induced_theory = CL_cruise_theory**2 / (np.pi * AR * eff)
        
        # Compare with identified parameter (which is multiplied by alpha^2)
        # So we need to account for alpha^2 scaling
        CD_induced_theory_param = CD_induced_theory / (alpha_cruise**2)
        
        print(f"\n   At α = 5°:")
        print(f"   - Theoretical CL: {CL_cruise_theory:.4f}")
        print(f"   - Theoretical CD_induced: {CD_induced_theory:.5f}")
        print(f"   - Theoretical CD_induced parameter: {CD_induced_theory_param:.4f}")
        print(f"   - Identified CD_induced parameter: {CD_induced:.4f}")
        
        if abs(CD_induced) > 0.01:
            ratio = CD_induced / CD_induced_theory_param
            print(f"   - Ratio (identified/theoretical): {ratio:.2f}")
            if 0.5 < ratio < 2.0:
                print(f"   ✓ Reasonable match (within 2x)")
            else:
                print(f"   ⚠ Large discrepancy - check data quality or model assumptions")
        else:
            print(f"   ⚠ CD_induced very small - may not be well identified")
        
        print("\n2. Stall Parameters Check")
        print("-" * 79)
        print(f"   Configured stall angle: {alpha_stall:.1f}°")
        print(f"   Identified Cema (pre-stall): {params.get('Cema', 0):.4f} /rad")
        print(f"   Identified Cema_stall (post-stall): {params.get('Cema_stall', 0):.4f} /rad")
        
        # Theoretical post-stall CL at stall angle
        alpha_s = np.deg2rad(alpha_stall)
        CL_poststall_theory = 2 * np.sign(alpha_s) * np.sin(alpha_s)**2 * np.cos(alpha_s)
        
        print(f"\n   At α = {alpha_stall:.1f}° (stall):")
        print(f"   - Theoretical CL (post-stall formula): {CL_poststall_theory:.4f}")
        print(f"   - Identified CL_poststall parameter: {CL_poststall:.4f}")
        
        if abs(CL_poststall) > 0.1:
            # Note: CL_poststall parameter is weighted by sigma, so direct comparison is complex
            print(f"   ℹ Post-stall parameter identified (requires σ-weighting for comparison)")
        
        print("\n3. Basic Aerodynamic Coefficient Ranges")
        print("-" * 79)
        
        # Typical ranges for fixed-wing aircraft
        checks = [
            ("CL0", CL0, (-0.2, 0.3), "Zero-lift coefficient"),
            ("CLa", CLa, (3.0, 7.0), "Lift curve slope (/rad)"),
            ("CD0", CD0, (0.015, 0.08), "Parasitic drag"),
            ("Cema", params.get('Cema', 0), (-2.0, -0.3), "Pitch moment slope (/rad)"),
            ("Cemq", params.get('Cemq', 0), (-1000, -100), "Pitch damping"),
            ("Cenb", params.get('Cenb', 0), (0.0, 0.01), "Directional stability"),
        ]
        
        for name, value, (min_val, max_val), description in checks:
            in_range = min_val <= value <= max_val
            status = "✓" if in_range else "⚠"
            print(f"   {status} {name:12s} = {value:8.4f}  (typical: [{min_val:.3f}, {max_val:.3f}]) - {description}")
            if not in_range and abs(value) > 0.001:
                if value < min_val:
                    print(f"      → Value is lower than typical range")
                else:
                    print(f"      → Value is higher than typical range")
        
        print("\n4. Control Surface Effectiveness")
        print("-" * 79)
        
        for ctrl_name in self.ctrl_surface_names:
            CL_ctrl = params.get(f'CL_{ctrl_name}', 0)
            Cem_ctrl = params.get(f'Cem_{ctrl_name}', 0)
            
            print(f"   {ctrl_name.capitalize()}:")
            print(f"   - CL_{ctrl_name}: {CL_ctrl:.5f} /deg")
            print(f"   - Cem_{ctrl_name}: {Cem_ctrl:.5f} /deg")
            
            # Elevator should have strong pitch effect
            if ctrl_name.lower() == 'elevator':
                if abs(Cem_ctrl) > 0.001:
                    print(f"   ✓ Elevator has pitch authority")
                else:
                    print(f"   ⚠ Elevator pitch authority seems low")
            
            # Rudder should have yaw effect
            elif ctrl_name.lower() == 'rudder':
                Cen_ctrl = params.get(f'Cen_{ctrl_name}', 0)
                print(f"   - Cen_{ctrl_name}: {Cen_ctrl:.5f} /deg")
                if abs(Cen_ctrl) > 0.001:
                    print(f"   ✓ Rudder has yaw authority")
                else:
                    print(f"   ⚠ Rudder yaw authority seems low")
            
            # Aileron should have roll effect  
            elif ctrl_name.lower() == 'aileron':
                Cell_ctrl = params.get(f'Cell_{ctrl_name}', 0)
                print(f"   - Cell_{ctrl_name}: {Cell_ctrl:.5f} /deg")
                if abs(Cell_ctrl) > 0.001:
                    print(f"   ✓ Aileron has roll authority")
                else:
                    print(f"   ⚠ Aileron roll authority seems low")
        
        print("\n5. Coupling Effects")
        print("-" * 79)
        
        # Check for significant coupling
        coupling_checks = [
            ("Aileron → Pitch", params.get('Cem_aileron', 0)),
            ("Rudder → Pitch", params.get('Cem_rudder', 0)),
            ("Elevator → Roll", params.get('Cell_elevator', 0)),
        ]
        
        for name, value in coupling_checks:
            if abs(value) > 0.01:
                print(f"   ℹ {name}: {value:.5f} (significant coupling detected)")
            else:
                print(f"   · {name}: {value:.5f} (minimal coupling)")
        
        print("\n" + "="*79)
        print("Note: These checks compare identified parameters with typical values.")
        print("Deviations may indicate: (1) unique aircraft characteristics,")
        print("(2) insufficient data excitation, or (3) model structure issues.")
        print("="*79 + "\n")
