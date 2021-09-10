__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.optimizers import OptimizerBaseTemplate
import cvxpy
import numpy as np
import warnings
from src.tools import math_tools
from sklearn.metrics import r2_score


class QPOptimizer(OptimizerBaseTemplate):

    def __init__(self, optimizer_config, param_name_list):
        super(QPOptimizer, self).__init__(optimizer_config, param_name_list)
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
        assert (len(param_bnd_keys) == self.n), "Optimizer needs exactly one bound per coefficient"
        self.G = []
        self.h = []
        self.fixed_coef_index_list = []
        self.fixed_value_coef_list = []
        for i in range(self.n):
            current_param = self.param_name_list[i]
            try:
                current_bnd_tuple = param_bounds[current_param]
            except IndexError:
                print("Can not find bounds for parameter " + current_param + " in config file.")
            print(current_param, current_bnd_tuple)
            if (current_bnd_tuple[0] == current_bnd_tuple[1]):
                self.fixed_coef_index_list.append(i)
                self.fixed_coef_value_list.append(current_bnd_tuple[0])
            else:
                self.G.append(-self.index_row(i))
                self.G.append(self.index_row(i))
                self.h.append(-current_bnd_tuple[0])
                self.h.append(current_bnd_tuple[1])
        self.G = np.array(self.G)
        self.h = np.array(self.h)

    def index_row(self, i):
        index_row = np.zeros(self.n)
        index_row[i] = 1
        return index_row

    def remove_fixed_coef_features(self, X, y):
        n_fixed_coef = len(self.fixed_coef_index_list)
        print("fixed values: ", self.fixed_value_coef_list)
        # remove elements starting from the back
        for i in range(n_fixed_coef):
            reversed_index = n_fixed_coef - i - 1
            X = np.delete(X, reversed_index, 1)
            y = y - X * self.fixed_value_coef_list[reversed_index]
        return X, y

    def insert_fixed_coefs(self, c_opt):
        print(c_opt)
        for i in range(len(self.fixed_coef_index_list)):
            c_opt.insert(self.fixed_coef_index_list[i], self.fixed_value_coef_list[i])
        print(c_opt)
        return c_opt

    def estimate_parameters(self, X, y):
        """
        Define and solve the CVXPY problem.

        min_c (X * c -y)^T * (X * c -y)
        s.t. G * c <= h
        """
        # remove fixed coefficients from problem formulation
        self.X, self.y = self.remove_fixed_coef_features(X, y)
        self.y = y
        c = cvxpy.Variable(self.n)
        cost = cvxpy.sum_squares(self.X @ c - self.y)
        self.prob = cvxpy.Problem(cvxpy.Minimize(cost), [self.G @ c <= self.h])
        self.prob.solve(verbose=True)
        self.c_opt = np.array(self.insert_fixed_coefs(c.value)).reshape((self.n, 1))
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
            "R2": float(r2_score(self.y, y_pred)),
            "Dual Variables": (self.prob.constraints[0].dual_value).tolist()
        }
        return metrics_dict
