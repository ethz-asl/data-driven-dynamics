"""
Full-fidelity AVL aerodynamic model with 36 + 6N parameters.

Implements complete AVL-style aerodynamics with proper feature matrix structure
for system identification. All parameters are affine (linear in the coefficients).

Copyright (c) 2024 Jaeyoung Lim, ETH Zurich ASL
License: BSD 3-Clause
"""

import numpy as np
from scipy.spatial.transform import Rotation
from progress.bar import Bar


class AVLModel:
    """
    Full AVL (Athena Vortex Lattice) aerodynamic model.
    
    Implements 36 + 6N parameters where N is the number of control surfaces:
    - 36 base aerodynamic parameters
    - 6 parameters per control surface (CD, CY, CL, Cell, Cem, Cen)
    
    All parameters are affine (linear) in the regression, computed from
    known flight states (velocities, angles, rates).
    """
    
    def __init__(self, config_dict):
        self.air_density = config_dict.get("air_density", 1.225)
        self.gravity = 9.81
        
        # Reference geometry
        self.area = config_dict["area"]
        self.chord = config_dict.get("chord", 0.0)
        self.span = config_dict.get("span", 0.0)
        self.AR = config_dict.get("AR", 0.0)
        self.eff = config_dict.get("eff", 0.95)
        
        # Calculate missing geometry
        if self.AR > 0 and self.span == 0:
            self.span = np.sqrt(self.area * self.AR)
        if self.chord == 0 and self.span > 0:
            self.chord = self.area / self.span
            
        # Stall parameters
        self.alpha_stall = config_dict.get("alpha_stall", np.deg2rad(15))
        self.M = config_dict.get("M", 15.0)
        
        # Flat plate drag parameters
        self.CD_fp_k1 = config_dict.get("CD_fp_k1", -3.0)
        self.CD_fp_k2 = config_dict.get("CD_fp_k2", 0.5)
        
        # Control surfaces
        self.num_ctrl_surfaces = config_dict.get("num_ctrl_surfaces", 0)
        self.ctrl_surface_names = config_dict.get("ctrl_surface_names", 
            [f"ctrl_{i}" for i in range(self.num_ctrl_surfaces)])

    def compute_force_features_single(self, v_airspeed, alpha, beta, omega, ctrl):
        """
        Compute force features in (3, n_params) matrix format.
        
        Each column represents one parameter's contribution to [Fx, Fy, Fz].
        All 36 + 6N base parameters are included.
        
        Returns:
            Flattened array in format [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, ...]
        """
        # Velocity and dynamic pressure
        v_x, v_y, v_z = v_airspeed[0], v_airspeed[1], v_airspeed[2]
        v_xz = np.sqrt(v_x**2 + v_z**2)
        q = 0.5 * self.air_density * v_xz**2
        half_rho_v = 0.5 * self.air_density * v_xz
        
        # Non-dimensional rates
        if v_xz > 1e-6:
            p_hat = omega[0] * self.span / (2 * v_xz)
            q_hat = omega[1] * self.chord / (2 * v_xz)
            r_hat = omega[2] * self.span / (2 * v_xz)
        else:
            p_hat = q_hat = r_hat = 0.0
        
        # Stall blending
        sigma = self._sigmoid_blend(alpha)
        sin_a, cos_a = np.sin(alpha), np.cos(alpha)
        sign_a = 1.0 if alpha >= 0 else -1.0
        
        # Flat plate drag for post-stall
        CD_fp = 2 / (1 + np.exp(self.CD_fp_k1 + self.CD_fp_k2 * max(self.AR, 1/self.AR)))
        
        # Rotation from stability to body frame
        R = Rotation.from_rotvec([0, alpha, 0]).as_matrix()
        
        # Build feature matrix: each column is one parameter's [Fx, Fy, Fz]
        features = []
        
        # ========== FORCE PARAMETERS (18 base) ==========
        
        # === Drag coefficients (6) ===
        # CD0 - parasitic drag
        features.append(R @ np.array([-q * self.area * (1 - sigma), 0, 0]))
        
        # CD_induced - induced drag (approximated as alpha^2)
        features.append(R @ np.array([-q * self.area * (1 - sigma) * alpha**2, 0, 0]))
        
        # CD_poststall - flat plate drag
        features.append(R @ np.array([-q * self.area * sigma * np.abs(CD_fp * (0.5 - 0.5*np.cos(2*alpha))), 0, 0]))
        
        # CDp - drag due to roll rate
        features.append(R @ np.array([-half_rho_v * self.area * self.span/2 * p_hat, 0, 0]))
        
        # CDq - drag due to pitch rate
        features.append(R @ np.array([-half_rho_v * self.area * self.chord/2 * q_hat, 0, 0]))
        
        # CDr - drag due to yaw rate
        features.append(R @ np.array([-half_rho_v * self.area * self.span/2 * r_hat, 0, 0]))
        
        # === Side force coefficients (5) ===
        # CYa - sideforce due to alpha
        features.append(R @ np.array([0, q * self.area * alpha, 0]))
        
        # CYb - sideforce due to sideslip
        features.append(R @ np.array([0, q * self.area * beta, 0]))
        
        # CYp - sideforce due to roll rate
        features.append(R @ np.array([0, half_rho_v * self.area * self.span/2 * p_hat, 0]))
        
        # CYq - sideforce due to pitch rate
        features.append(R @ np.array([0, half_rho_v * self.area * self.chord/2 * q_hat, 0]))
        
        # CYr - sideforce due to yaw rate
        features.append(R @ np.array([0, half_rho_v * self.area * self.span/2 * r_hat, 0]))
        
        # === Lift coefficients (7) ===
        # CL0 - zero-alpha lift
        features.append(R @ np.array([0, 0, -q * self.area * (1 - sigma)]))
        
        # CLa - lift curve slope
        features.append(R @ np.array([0, 0, -q * self.area * (1 - sigma) * alpha]))
        
        # CL_poststall - post-stall lift
        features.append(R @ np.array([0, 0, -q * self.area * sigma * sign_a * sin_a**2 * cos_a]))
        
        # CLb - lift due to sideslip
        features.append(R @ np.array([0, 0, -q * self.area * beta]))
        
        # CLp - lift due to roll rate
        features.append(R @ np.array([0, 0, -half_rho_v * self.area * self.span/2 * p_hat]))
        
        # CLq - lift due to pitch rate
        features.append(R @ np.array([0, 0, -half_rho_v * self.area * self.chord/2 * q_hat]))
        
        # CLr - lift due to yaw rate
        features.append(R @ np.array([0, 0, -half_rho_v * self.area * self.span/2 * r_hat]))
        
        # === Control surface force effects (3 per surface) ===
        for i in range(self.num_ctrl_surfaces):
            delta = np.rad2deg(ctrl[i]) if i < len(ctrl) else 0.0
            
            # CD_ctrl - drag due to control deflection
            features.append(R @ np.array([-q * self.area * delta * 0.001, 0, 0]))
            
            # CY_ctrl - sideforce due to control deflection
            features.append(R @ np.array([0, q * self.area * delta * 0.001, 0]))
            
            # CL_ctrl - lift due to control deflection
            features.append(R @ np.array([0, 0, -q * self.area * delta * 0.01]))
        
        # Stack into (3, n_params) matrix and flatten
        X = np.column_stack(features)
        return X.T.flatten()

    def compute_moment_features_single(self, v_airspeed, alpha, beta, omega, ctrl):
        """
        Compute moment features in (3, n_params) matrix format.
        
        Each column represents one parameter's contribution to [Mx, My, Mz].
        All 19 + 3N base parameters are included.
        
        Returns:
            Flattened array in format [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, ...]
        """
        v_xz = np.sqrt(v_airspeed[0]**2 + v_airspeed[2]**2)
        q = 0.5 * self.air_density * v_xz**2
        half_rho_v = 0.5 * self.air_density * v_xz
        
        # Non-dimensional rates
        if v_xz > 1e-6:
            p_hat = omega[0] * self.span / (2 * v_xz)
            q_hat = omega[1] * self.chord / (2 * v_xz)
            r_hat = omega[2] * self.span / (2 * v_xz)
        else:
            p_hat = q_hat = r_hat = 0.0
        
        # Stall check for pitch moment
        in_stall_pos = alpha > self.alpha_stall
        in_stall_neg = alpha < -self.alpha_stall
        
        features = []
        
        # ========== MOMENT PARAMETERS (19 base + 3N ctrl) ==========
        
        # === Roll moment (Cell) - 5 params ===
        # Cella - roll due to alpha
        features.append(np.array([q * self.area * self.span * alpha, 0, 0]))
        
        # Cellb - roll due to beta (dihedral effect)
        features.append(np.array([q * self.area * self.span * beta, 0, 0]))
        
        # Cellp - roll damping
        features.append(np.array([half_rho_v * self.area * self.span * self.span/2 * p_hat, 0, 0]))
        
        # Cellq - roll due to pitch rate
        features.append(np.array([half_rho_v * self.area * self.span * self.chord/2 * q_hat, 0, 0]))
        
        # Cellr - roll due to yaw rate
        features.append(np.array([half_rho_v * self.area * self.span * self.span/2 * r_hat, 0, 0]))
        
        # === Pitch moment (Cem) - 8 params ===
        # Cem0 - zero-alpha pitch moment
        features.append(np.array([0, q * self.area * self.chord, 0]))
        
        # Cema - pitch moment slope (pre-stall)
        if not in_stall_pos and not in_stall_neg:
            features.append(np.array([0, q * self.area * self.chord * alpha, 0]))
        elif in_stall_pos:
            features.append(np.array([0, q * self.area * self.chord * self.alpha_stall, 0]))
        else:
            features.append(np.array([0, -q * self.area * self.chord * self.alpha_stall, 0]))
        
        # Cema_stall - pitch moment slope (post-stall)
        if in_stall_pos:
            features.append(np.array([0, q * self.area * self.chord * (alpha - self.alpha_stall), 0]))
        elif in_stall_neg:
            features.append(np.array([0, q * self.area * self.chord * (alpha + self.alpha_stall), 0]))
        else:
            features.append(np.zeros(3))
        
        # Cemb - pitch moment due to sideslip
        features.append(np.array([0, q * self.area * self.chord * beta, 0]))
        
        # Cemp - pitch moment due to roll rate
        features.append(np.array([0, half_rho_v * self.area * self.chord * self.span/2 * p_hat, 0]))
        
        # Cemq - pitch damping
        features.append(np.array([0, half_rho_v * self.area * self.chord * self.chord/2 * q_hat, 0]))
        
        # Cemr - pitch moment due to yaw rate
        features.append(np.array([0, half_rho_v * self.area * self.chord * self.span/2 * r_hat, 0]))
        
        # === Yaw moment (Cen) - 6 params ===
        # Cena - yaw due to alpha
        features.append(np.array([0, 0, q * self.area * self.span * alpha]))
        
        # Cenb - yaw due to beta (directional stability)
        features.append(np.array([0, 0, q * self.area * self.span * beta]))
        
        # Cenp - yaw due to roll rate
        features.append(np.array([0, 0, half_rho_v * self.area * self.span * self.span/2 * p_hat]))
        
        # Cenq - yaw due to pitch rate
        features.append(np.array([0, 0, half_rho_v * self.area * self.span * self.chord/2 * q_hat]))
        
        # Cenr - yaw damping
        features.append(np.array([0, 0, half_rho_v * self.area * self.span * self.span/2 * r_hat]))
        
        # === Control surface moment effects (3 per surface) ===
        for i in range(self.num_ctrl_surfaces):
            delta = np.rad2deg(ctrl[i]) if i < len(ctrl) else 0.0
            
            # Cell_ctrl - roll moment due to control
            features.append(np.array([q * self.area * self.span * delta * 0.001, 0, 0]))
            
            # Cem_ctrl - pitch moment due to control
            features.append(np.array([0, q * self.area * self.chord * delta * 0.01, 0]))
            
            # Cen_ctrl - yaw moment due to control
            features.append(np.array([0, 0, q * self.area * self.span * delta * 0.001]))
        
        # Stack and flatten
        X = np.column_stack(features)
        return X.T.flatten()

    def compute_aero_force_features(self, v_airspeed_mat, alpha_vec, beta_vec, omega_mat, ctrl_mat=None):
        """Batch compute force features for all timesteps."""
        n_samples = v_airspeed_mat.shape[0]
        
        if ctrl_mat is None:
            ctrl_mat = np.zeros((n_samples, self.num_ctrl_surfaces))
        
        print("Computing AVL aerodynamic force features (full model)...")
        features_bar = Bar("Force Features", max=n_samples)
        
        X_aero = self.compute_force_features_single(
            v_airspeed_mat[0, :], alpha_vec[0], beta_vec[0],
            omega_mat[0, :], ctrl_mat[0, :]
        )
        
        for i in range(1, n_samples):
            X_curr = self.compute_force_features_single(
                v_airspeed_mat[i, :], alpha_vec[i], beta_vec[i],
                omega_mat[i, :], ctrl_mat[i, :]
            )
            X_aero = np.vstack((X_aero, X_curr))
            features_bar.next()
        features_bar.finish()
        
        coef_dict, col_names = self._generate_force_coef_names()
        return X_aero, coef_dict, col_names

    def compute_aero_moment_features(self, v_airspeed_mat, alpha_vec, beta_vec, omega_mat, ctrl_mat=None):
        """Batch compute moment features for all timesteps."""
        n_samples = v_airspeed_mat.shape[0]
        
        if ctrl_mat is None:
            ctrl_mat = np.zeros((n_samples, self.num_ctrl_surfaces))
        
        print("Computing AVL aerodynamic moment features (full model)...")
        features_bar = Bar("Moment Features", max=n_samples)
        
        X_aero = self.compute_moment_features_single(
            v_airspeed_mat[0, :], alpha_vec[0], beta_vec[0],
            omega_mat[0, :], ctrl_mat[0, :]
        )
        
        for i in range(1, n_samples):
            X_curr = self.compute_moment_features_single(
                v_airspeed_mat[i, :], alpha_vec[i], beta_vec[i],
                omega_mat[i, :], ctrl_mat[i, :]
            )
            X_aero = np.vstack((X_aero, X_curr))
            features_bar.next()
        features_bar.finish()
        
        coef_dict, col_names = self._generate_moment_coef_names()
        return X_aero, coef_dict, col_names

    def _sigmoid_blend(self, alpha):
        """Sigmoid blending function for pre/post-stall transition."""
        return (1 + np.exp(-self.M * (alpha - self.alpha_stall)) + 
                np.exp(self.M * (alpha + self.alpha_stall))) / (
                (1 + np.exp(-self.M * (alpha - self.alpha_stall))) * 
                (1 + np.exp(self.M * (alpha + self.alpha_stall))))

    def _generate_force_coef_names(self):
        """Generate coefficient names for force features."""
        base_params = [
            "CD0", "CD_induced", "CD_poststall", "CDp", "CDq", "CDr",
            "CYa", "CYb", "CYp", "CYq", "CYr",
            "CL0", "CLa", "CL_poststall", "CLb", "CLp", "CLq", "CLr"
        ]
        
        # Add control surface parameters (3 per surface)
        for name in self.ctrl_surface_names:
            base_params.extend([f"CD_{name}", f"CY_{name}", f"CL_{name}"])
        
        # Generate column names: [p0_x, p0_y, p0_z, p1_x, p1_y, p1_z, ...]
        col_names = []
        for param in base_params:
            col_names.extend([f"{param}_x", f"{param}_y", f"{param}_z"])
        
        # Coefficient dictionary
        coef_dict = {}
        for param in base_params:
            coef_dict[param] = {"lin": {
                "x": f"{param}_x",
                "y": f"{param}_y",
                "z": f"{param}_z"
            }}
        
        return coef_dict, col_names

    def _generate_moment_coef_names(self):
        """Generate coefficient names for moment features."""
        base_params = [
            "Cella", "Cellb", "Cellp", "Cellq", "Cellr",
            "Cem0", "Cema", "Cema_stall", "Cemb", "Cemp", "Cemq", "Cemr",
            "Cena", "Cenb", "Cenp", "Cenq", "Cenr"
        ]
        
        # Add control surface parameters (3 per surface)
        for name in self.ctrl_surface_names:
            base_params.extend([f"Cell_{name}", f"Cem_{name}", f"Cen_{name}"])
        
        col_names = []
        for param in base_params:
            col_names.extend([f"{param}_x", f"{param}_y", f"{param}_z"])
        
        coef_dict = {}
        for param in base_params:
            coef_dict[param] = {"rot": {
                "x": f"{param}_x",
                "y": f"{param}_y",
                "z": f"{param}_z"
            }}
        
        return coef_dict, col_names
