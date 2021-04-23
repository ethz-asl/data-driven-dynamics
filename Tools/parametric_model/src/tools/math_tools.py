__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np


def sym_sigmoid(x, x_offset=0, scale_fac=1):
    # computes a logistic sigmoid function which is symmetric
    # around zero and crosses the 0.5 mark at +- x_offset
    y = 1 - (math.exp(scale_fac*(x+x_offset))) / \
        ((1+math.exp(scale_fac*(x-x_offset)))*(1+math.exp(scale_fac*(x+x_offset))))
    return y


def rmse_between_numpy_arrays(np_array1, np_array2):
    difference_array = np.subtract(np_array1, np_array2)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return math.sqrt(mse)
