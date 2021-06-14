__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import math
import numpy as np
import matplotlib.pyplot as plt


def cropped_sym_sigmoid(x, x_offset=0, scale_fac=30):
    y = sym_sigmoid(x, x_offset, scale_fac)
    if y < 0.001:
        y = 0
    elif y > 0.999:
        y = 1
    return y


def sym_sigmoid(x, x_offset=0, scale_fac=30):
    # computes a logistic sigmoid function which is symmetric
    # around zero and crosses the 0.5 mark at +- x_offset
    y = 1 - (math.exp(scale_fac*(x+x_offset))) / \
        ((1+math.exp(scale_fac*(x-x_offset)))*(1+math.exp(scale_fac*(x+x_offset))))
    return y


def plot_sym_sigmoid(scale_fac, x_offset=0.26, x_range=90):
    N = x_range*2+1
    x = np.linspace(-x_range, x_range, N)
    x_rad = x*math.pi/180.0
    y = np.zeros(N)
    for i in range(N):
        y[i] = sym_sigmoid(x_rad[i], x_offset, scale_fac)
    plt.xlabel('Angle of Attack (deg)')
    plt.ylabel(r'$\sigma \ (\alpha)$')
    plt.plot(x, y)
    plt.show()


def rmse_between_numpy_arrays(np_array1, np_array2):
    difference_array = np.subtract(np_array1, np_array2)
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return math.sqrt(mse)


if __name__ == "__main__":
    # run this script to find suitable values for the scale_factor of the symmetric sigmoid function
    plot_sym_sigmoid(50, x_range=35)
