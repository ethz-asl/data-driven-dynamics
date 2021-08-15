__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

from src.optimizers import OptimizerBaseTemplate
from sklearn.linear_model import LinearRegression


class LinearRegressor(OptimizerBaseTemplate):

    def __init__(self, optimizer_config):
        super(LinearRegressor, self).__init__(optimizer_config)
        self.reg = LinearRegression(fit_intercept=False)

    def estimate_parameters(self, X, y):
        # Estimate parameters c such that X * c = y
        self.X = X
        self.y = y
        self.reg.fit(self.X, self.y)
        self.estimation_completed = True

    def get_optimization_parameters(self):
        self.check_estimation_completed()
        return list(self.reg.coef_)

    def predict(self, X_pred):
        self.check_estimation_completed()
        return self.reg.predict(X_pred)

    def compute_optimization_metrics(self):
        self.check_estimation_completed()
        metrics_dict = {"R2": float(self.reg.score(self.X, self.y))}
        return metrics_dict
