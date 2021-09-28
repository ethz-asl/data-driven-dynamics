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

import numpy as np
import csv
import h5py
import time

from pyulog.core import ULog

DOT_THRESHOLD = 0.9995


def quaternion_to_rotation_matrix(q):
    # Construct a rotation matrix from quaternions
    qr, qi, qj, qk = q[0], q[1], q[2], q[3]
    R = np.array([[1-2*(qj**2+qk**2), 2*(qi*qj-qk*qr), 2*(qi*qk + qj*qr)],
                  [2*(qi*qj + qk*qr), 1-2*(qi**2+qk**2),  2*(qk*qj-qi*qr)],
                  [2*(qi*qk - qj*qr), 2*(qj*qk + qi*qr), 1-2*(qi**2+qj**2)]])
    return R


def slerp(v0, v1, t_array):
    # This is a quaternion interpolation method
    # >>> slerp([1,0,0,0],[0,0,0,1],np.arange(0,1,0.001))
    # t_array are time indexes for the interpolation,
    # where t \in [0, 1], and v0 is at t=0, v1 at t=1
    # From Wikipedia: https://en.wikipedia.org/wiki/Slerp
    t_array = np.array(t_array)
    v0 = np.array(v0)
    v1 = np.array(v1)
    dot = np.sum(v0 * v1)
    if (dot < 0.0):
        v1 = -v1
        dot = -dot
    if (dot > DOT_THRESHOLD):
        result = v0[np.newaxis, :] + t_array[:,
                                             np.newaxis] * (v1 - v0)[np.newaxis, :]
        result = result / np.linalg.norm(result)
        return result
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t_array
    sin_theta = np.sin(theta)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return (s0[:, np.newaxis] * v0[np.newaxis, :]) + (s1[:, np.newaxis] * v1[np.newaxis, :])
