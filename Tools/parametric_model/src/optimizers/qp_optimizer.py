__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.optimizers import OptimizerBaseTemplate
import cvxpy
import numpy as np
import warnings
from src.tools import math_tools


class QPOptimizer(OptimizerBaseTemplate):

    def __init__(self, optimizer_config, param_name_list):
        super(QPOptimizer, self).__init__(optimizer_config, param_name_list)
        self.n = len(param_name_list)
        if "parameter_bounds" in self.config:
            self.__compute_ineq_contraints(param_name_list)
        else:
            warnings.warn("You have selected the QPOptimizer for linear models with \
                          bound constrains but have not specified any bounds. \
                          Consider switching to LinearRegression for unconstraint parameter estimation.")
            self.G = np.zeros(
                (1, len(param_name_list)))
            self.h = np.zeros(1)

    def __compute_ineq_contraints(self, param_name_list):
        param_bounds = self.config["parameter_bounds"]
        param_bnd_keys = list(param_bounds.keys())
        self.G = np.zeros(
            (2*len(param_bnd_keys), len(param_name_list)))
        self.h = np.zeros(2*len(param_bnd_keys))
        for i in range(len(param_bnd_keys)):
            current_param = param_bnd_keys[i]
            try:
                curr_param_index = param_name_list.index(current_param)
            except IndexError:
                print("Config file specifies bounds for " + current_param +
                      "which is not a parameter for the chosen model.")
                print("Valid parameters for this model are: ", param_name_list)
            current_bnd_tuple = param_bounds[current_param]
            self.G[2*i, curr_param_index] = -1
            self.G[2*i+1, curr_param_index] = 1
            self.h[2*i] = -current_bnd_tuple[0]
            self.h[2*i+1] = current_bnd_tuple[1]

    def estimate_parameters(self, X, y):
        """
        Define and solve the CVXPY problem.

        min_c (X * c -y)^T * (X * c -y)
        s.t. G * c <= h
        """
        self.X = X
        self.y = y
        c = cvxpy.Variable(self.n)
        cost = cvxpy.sum_squares(self.X @ c - self.y)
        self.prob = cvxpy.Problem(cvxpy.Minimize(cost), [self.G @ c <= self.h])
        self.prob.solve(verbose=True)
        self.c_opt = np.array(c.value).reshape((self.n, 1))
        print(self.c_opt)
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
        metrics_dict = {"Dual Variables": (self.prob.constraints[0].dual_value).tolist(),
                        "RMSE": math_tools.rmse_between_numpy_arrays(y_pred, self.y)
                        }
        from src.models.model_plots import model_plots
        print(y_pred.shape)
        print(self.y.shape)
        # n = list(range(y_pred.shape[0]/3))
        # print(n)
        # model_plots.plot_accel_predeictions(
        #     self.y, y_pred, n)

        return metrics_dict
