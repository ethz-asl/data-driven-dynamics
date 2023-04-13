import os
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

The model class contains properties shared between all models and shgall simplyfy automated checks and the later
export to a sitl gazebo model by providing a unified interface for all models. """

__author__ = "Manuel Yves Galliker, Julius Schlapbach"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"


class AutomaticDataSelector():
    def __init__(self, data_df):
        self.data_df = data_df

    def select_dataframes(self, ratio=10):
        idx = self.data_df.sort_values(
            by=["fisher_information_force"]).index[0:self.data_df.shape[0]*10//100]
        idx = idx.append(self.data_df.sort_values(
            by=["fisher_information_rot"]).index[0:self.data_df.shape[0]*10//100])
        idx = idx.unique()
        idx = idx.sort_values()
        self.data_df = self.data_df.loc[idx]
        self.data_df.reset_index(drop=True)
        return self.data_df
