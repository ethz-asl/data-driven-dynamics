"""
 *
 * Copyright (c) 2021 Manuel Galliker
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

import math
import numpy as np
import matplotlib.pyplot as plt


def cropped_sym_sigmoid(x, x_offset=0, scale_fac=30):
    y = sym_sigmoid(x, x_offset, scale_fac)
    if y < 0.025:
        y = 0
    elif y > 0.975:
        y = 1
    return y


def sym_sigmoid(x, x_offset=0, scale_fac=30):
    # computes a logistic sigmoid function which is symmetric
    # around zero and crosses the 0.5 mark at +- x_offset
    y = 1 - (math.exp(scale_fac*(x+x_offset))) / \
        ((1+math.exp(scale_fac*(x-x_offset)))*(1+math.exp(scale_fac*(x+x_offset))))
    return y


def plot_sym_sigmoid(scale_fac, x_offset=0.35, x_range=90):
    N = x_range*2+1
    x = np.linspace(-x_range, x_range, N)
    x_rad = x*math.pi/180.0
    y = np.zeros(N)
    for i in range(N):
        y[i] = sym_sigmoid(x_rad[i], x_offset, scale_fac)
    plt.xlabel('Angle of Attack (deg)')
    plt.plot(x, y)
    plt.show()


def rmse_between_numpy_arrays(np_array1, np_array2):
    difference_array = np.subtract(np_array1.flatten(), np_array2.flatten())
    squared_array = np.square(difference_array)
    mse = squared_array.mean()
    return math.sqrt(mse)


if __name__ == "__main__":
    # run this script to find suitable values for the scale_factor of the symmetric sigmoid function
    plot_sym_sigmoid(30, x_range=180)
