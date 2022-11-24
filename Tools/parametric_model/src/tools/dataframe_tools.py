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

__author__ = "Manuel Yves Galliker"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
from src.tools.ulog_tools import pandas_from_topic
from src.tools.quat_utils import slerp


def compute_flight_time(act_df, pwm_threshold=None, control_threshold=None):
    """This function computes the flight time by:
    Option 1: listen to vehicle_land_detected/landed
    Option 2: user defined start/end time stamp (micro seconds) 
    """

    # # Option 1:
    # print("act_df: ", act_df)
    # act_df_crp = act_df[act_df.iloc[:, 4] < 1] # take part where landed is 0
    # print("act_df_crp after selection: ", act_df_crp)

    # Option 2:
    # print("act_df: ", act_df)
    act_df_crp = act_df[act_df.iloc[:, 0] > 405000000]
    act_df_crp = act_df_crp[act_df_crp.iloc[:, 0] < 430000000]
    # print("act_df_crp after selection: ", act_df_crp)

    t_start = act_df_crp.iloc[1, 0]
    t_end = act_df_crp.iloc[(act_df_crp.shape[0]-1), 0]
    flight_time = {"t_start": t_start, "t_end": t_end}
    return flight_time


def moving_average(x, w=7):
    return np.convolve(x, np.ones(w), 'valid') / w


def filter_df(data_df, w=11):
    data_np = data_df.to_numpy()
    column_list = data_df.columns
    new_df = pd.DataFrame()
    for i in range(data_np.shape[1]):
        new_df[column_list[i]] = moving_average(data_np[:, i])
    return new_df


def resample_dataframe_list(df_list, time_window=None, f_des=100.0, slerp_enabled=False, filter=True):
    """create a single dataframe by resampling all dataframes to f_des [Hz]

    Inputs:     df_list : List of ulog topic dataframes to resample
                t_start : Start time in us
                t_end   : End time in us
                f_des   : Desired frequency of resampled data   
    """
    if time_window is None:
        # select full ulog time range
        df = df_list[0]
        timestamp_list = df["timestamp"].to_numpy()
        t_start = timestamp_list[0]
        t_end = timestamp_list[-1]

    else:
        t_start = time_window["t_start"]
        t_end = time_window["t_end"]

    # compute desired Period in us to be persistent with ulog timestamps
    assert f_des > 0, 'Desired frequency must be greater than 0'
    T_des = 1000000.0/f_des

    n_samples = int((t_end-t_start)/T_des)
    res_df = pd.DataFrame()
    new_t_list = np.arange(t_start, t_end, T_des)
    for df in df_list:
        df = filter_df(df)
        df_end = df["timestamp"].iloc[[-1]].to_numpy()
        if df_end < t_end:
            t_end = int(df_end)

    for df in df_list:
        # use slerp interpolation for quaternions
        # add a better criteria than the exact naming at a later point.
        if 'q0' in df and slerp_enabled:
            q_mat = slerp_interpolate_from_df(df, new_t_list[0])

            for i in range(1, len(new_t_list)):
                q_new = slerp_interpolate_from_df(df, new_t_list[i])
                q_mat = np.vstack((q_mat, q_new))
            attitude_col_names = list(df.columns)
            attitude_col_names.remove("timestamp")
            new_df = pd.DataFrame(q_mat, columns=attitude_col_names)

        else:
            new_df = pd.DataFrame()
            for col in df:
                new_df[col] = np.interp(new_t_list, df.timestamp, df[col])

        res_df = pd.concat([res_df, new_df], axis=1)
        res_df = res_df.loc[:, ~res_df.columns.duplicated()]

    return res_df


def slerp_interpolate_from_df(df, new_t):
    df_sort = df.iloc[(df['timestamp']-new_t).abs().argsort()[:2]]
    df_timestamps = df_sort['timestamp'].values.tolist()
    t_ratio = (new_t - df_timestamps[0]) / \
        (df_timestamps[1] - df_timestamps[0])
    df_sort = df_sort.drop(columns=['timestamp'])

    q_new = slerp(df_sort.iloc[0, :].to_numpy(
    ), df_sort.iloc[1, :].to_numpy(), np.array([t_ratio]))
    return q_new


def crop_df(df, t_start, t_end):
    """ crop df to contain 1 elemnt before t_start and one after t_end.
    This way it is easy to interpolate the data between start and end time. """
    df_start = df[df.timestamp <= t_start].iloc[[-1]]
    df_end = df[df.timestamp >= t_end].iloc[[0]]

    df = df[df.timestamp >= int(df_start.timestamp.to_numpy())]
    df = df[df.timestamp <= int(df_end.timestamp.to_numpy())]
    return df
