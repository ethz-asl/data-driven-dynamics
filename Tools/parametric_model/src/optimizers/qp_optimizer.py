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

from src.optimizers import OptimizerBaseTemplate
import cvxpy
import numpy as np
import pandas as pd
import warnings
from src.tools import math_tools
from sklearn.metrics import r2_score


class QPOptimizer(OptimizerBaseTemplate):

    def __init__(self, optimizer_config, param_name_list, verbose=False):
        super(QPOptimizer, self).__init__(optimizer_config, param_name_list)
        print("Define and solve problem:")
        print("min_c (X * c -y)^T * (X * c -y)")
        print(" s.t. G * c <= h")
        print("Initialized with the following coefficients: ")
        print(param_name_list)
        self.verbose = verbose
        self.n = len(param_name_list)
        self.param_name_list = param_name_list
        if "parameter_bounds" in self.config:
            self.__compute_ineq_contraints()
        else:
            warnings.warn("You have selected the QPOptimizer for linear models with \
                          bound constrains but have not specified any bounds. \
                          Consider switching to LinearRegression for unconstraint parameter estimation.")
            self.G = np.zeros(
                (1, len(param_name_list)))
            self.h = np.zeros(1)

    def __compute_ineq_contraints(self):
        param_bounds = self.config["parameter_bounds"]
        param_bnd_keys = list(param_bounds.keys())
        # assert (len(param_bnd_keys) ==
        #         self.n), "Optimizer needs exactly one bound per coefficient"
        self.G = []
        self.h = []
        self.fixed_coef_index_list = []
        self.fixed_coef_value_list = []
        self.optimization_coef_list = []
        self.fixed_coef_list = []
        for i in range(self.n):
            current_param = self.param_name_list[i]
            try:
                current_bnd_tuple = param_bounds[current_param]
            except IndexError:
                print("Can not find bounds for parameter " +
                      current_param + " in config file.")
            if (current_bnd_tuple[0] == current_bnd_tuple[1]):
                self.fixed_coef_index_list.append(i)
                self.fixed_coef_value_list.append(current_bnd_tuple[0])
                self.fixed_coef_list.append(current_param)
            else:
                self.G.append(-self.index_row(i))
                self.G.append(self.index_row(i))
                self.h.append(-current_bnd_tuple[0])
                self.h.append(current_bnd_tuple[1])
                self.optimization_coef_list.append(current_param)
        self.G = np.array(self.G)
        self.h = np.array(self.h)
        self.n_fixed_coef = len(self.fixed_coef_index_list)
        self.n_opt_coef = self.n - self.n_fixed_coef
        for i in range(self.n_fixed_coef):
            reversed_index = self.n_fixed_coef - i - 1
            self.G = np.delete(
                self.G, self.fixed_coef_index_list[reversed_index], 1)

        print("Fixed Coefficients: Value")
        for fixed_coef in self.fixed_coef_list:
            print(fixed_coef + ": ", param_bounds[fixed_coef][0])
        print("-------------------------------------------------------------------------------")
        print("Bounded Coefficients: (Min Value, Max Value)")
        for opt_coef in self.optimization_coef_list:
            print(opt_coef + ": ", param_bounds[opt_coef])
        if self.verbose:
            print(
                "-------------------------------------------------------------------------------")
            print(
                "                           Constraints matrices                                ")
            print(
                "-------------------------------------------------------------------------------")
            print(self.G)
            print(self.h)

    def index_row(self, i):
        index_row = np.zeros(self.n)
        index_row[i] = 1
        return index_row

    def remove_fixed_coef_features(self, X, y):
        # remove elements starting from the back
        for i in range(self.n_fixed_coef):
            reversed_index = self.n_fixed_coef - i - 1
            coef_index = self.fixed_coef_index_list[reversed_index]
            print(self.fixed_coef_index_list)
            print(coef_index)
            print(self.n)
            print(X.shape)
            y = y - (X[:, coef_index] *
                     self.fixed_coef_value_list[reversed_index]).flatten()
            X = np.delete(X, coef_index, 1)
        return X, y

    def insert_fixed_coefs(self, c_opt):
        c_opt = list(c_opt)
        for i in range(len(self.fixed_coef_index_list)):
            c_opt.insert(
                self.fixed_coef_index_list[i], self.fixed_coef_value_list[i])
        return c_opt

    def estimate_parameters(self, X, y):
        """
        Define and solve the CVXPY problem.

        min_c (X * c -y)^T * (X * c -y)
        s.t. G * c <= h
        """
        # remove fixed coefficients from problem formulation
        self.X = X
        self.y = y
        self.X_reduced, self.y_reduced = self.remove_fixed_coef_features(X, y)
        self.y = y
        c = cvxpy.Variable(self.n_opt_coef)
        cost = cvxpy.sum_squares(self.X_reduced @ c - self.y_reduced)
        self.prob = cvxpy.Problem(cvxpy.Minimize(cost), [self.G @ c <= self.h])
        self.prob.solve(verbose=self.verbose)
        self.c_opt = np.array(self.insert_fixed_coefs(
            c.value)).reshape((self.n, 1))
        self.estimation_completed = True

    def get_optimization_parameters(self):
        self.check_estimation_completed()
        return list(self.c_opt)

    def predict(self, X_pred):
        self.check_estimation_completed()
        y_pred = np.matmul(X_pred, self.c_opt)
        return y_pred.flatten()

    def compute_optimization_metrics(self):
        self.check_estimation_completed()
        y_pred = self.predict(self.X)
        metrics_dict = {
            "RMSE": math_tools.rmse_between_numpy_arrays(y_pred, self.y),
            "R2": float(r2_score(self.y, y_pred))
        }
        if self.verbose:
            metrics_dict["Dual Variables"] = (
                self.prob.constraints[0].dual_value).tolist()

        return metrics_dict
