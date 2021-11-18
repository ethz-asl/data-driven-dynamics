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

__author__ = "Manuel Yves Galliker"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

from src.optimizers import OptimizerBaseTemplate
from sklearn.linear_model import LinearRegression
from src.tools import math_tools


class LinearRegressor(OptimizerBaseTemplate):

    def __init__(self, optimizer_config, param_name_list):
        super(LinearRegressor, self).__init__(
            optimizer_config, param_name_list)
        print("Define and solve problem:")
        print("min_c (X * c -y)^T * (X * c -y)")
        self.reg = LinearRegression(fit_intercept=False)

    def estimate_parameters(self, X, y):
        # Estimate parameters c such that X * c = y
        self.X = X
        self.y = y
        self.check_features()
        self.reg.fit(self.X, self.y)
        self.estimation_completed = True

    def get_optimization_parameters(self):
        self.check_estimation_completed()
        return list(self.reg.coef_)

    def set_optimal_coefficients(self, c_opt, X, y):
        self.X = X
        self.y = y
        self.reg.coef_ = c_opt
        self.estimation_completed = True

    def predict(self, X_pred):
        self.check_estimation_completed()
        return self.reg.predict(X_pred)

    def compute_optimization_metrics(self):
        self.check_estimation_completed()
        y_pred = self.predict(self.X)
        metrics_dict = {"R2": float(self.reg.score(self.X, self.y)),
                        "RMSE": math_tools.rmse_between_numpy_arrays(y_pred, self.y)}
        return metrics_dict
