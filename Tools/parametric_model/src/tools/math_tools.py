__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math


def sym_sigmoid(x, x_offset=0, scale_fac=1):
    # computes a logistic sigmoid function which is symmetric
    # around zero and crosses the 0.5 mark at +- x_offset
    y = 1 - (math.exp(scale_fac*(x+x_offset))) / \
        ((1+math.exp(scale_fac*(x-x_offset)))*(1+math.exp(scale_fac*(x+x_offset))))
    return y
