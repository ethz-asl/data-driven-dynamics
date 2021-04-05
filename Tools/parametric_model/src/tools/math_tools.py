__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math


def symmetric_logistic_sigmoid(x, x_offset=0, scale_fac=1):
    y = (1+math.exp(scale_fac(x-x_offset)) + math.exp(scale_fac(x+x_offset))) / \
        ((1+math.exp(scale_fac(x-x_offset)))*(1+math.exp(scale_fac(x+x_offset))))
    return y
