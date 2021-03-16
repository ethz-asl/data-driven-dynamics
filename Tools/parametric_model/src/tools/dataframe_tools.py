__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
from src.tools import pandas_from_topic


def resample_dataframes(df_list, t_start, t_end, f_des):
    """create a single dataframe by resampling all dataframes to f_des [Hz]

    Inputs:     df_list : List of ulog topic dataframes to resample
                t_start : Start time in us
                t_end   : End time in us
                f_des   : Desired frequency of resampled data   
    """

    # compute desired Period in us to be persistent with ulog timestamps
    T_des = 1000000.0/f_des
    n_samples = int((t_end-t_start)/T_des)
    for df in df_list:
        df = crop_df(df, t_start, t_end)
        new_df = pd.DataFrame(
            np.zeros((n_samples, df.shape[1])), columns=df.columns)
        for n in range(n_samples):
            t_curr = t_start + n*T_des
            new_df.iloc[[n]] = _interpolate_to_timestamp(df, t_curr)
        print(new_df)


def _interpolate_to_timestamp(df, timestamp):
    """ extracts the two rows of df with timestamp closest to a specified 
    timestamp and linearly interpolates its values."""

    # extract rows before and after timestamp
    r_1 = df[df.timestamp <= timestamp].iloc[[-1]]
    r_2 = df[df.timestamp >= timestamp].iloc[[0]]

    if (int(r_1.timestamp.to_numpy()) == int(r_2.timestamp.to_numpy())):
        r_n = r_1
    else:
        r_n = pd.DataFrame(
            np.zeros((1, df.shape[1])), columns=df.columns)
        r_n.timestamp = timestamp
        for r in range(1, df.shape[1]):
            r_n.iloc[0, r] = r_1.iloc[0, r] + \
                (timestamp - r_1.iloc[0, 0]) * \
                (r_2.iloc[0, r] - r_1.iloc[0, r]) / \
                (r_2.iloc[0, 0] - r_1.iloc[0, 0])

    return r_n


def crop_df(df, t_start, t_end):
    # crop df to contain 1 elemnt before t_start and one after t_end
    df_start = df[df.timestamp <= t_start].iloc[[-1]]
    df_end = df[df.timestamp >= t_end].iloc[[0]]

    df = df[df.timestamp >= int(df_start.timestamp.to_numpy())]
    df = df[df.timestamp <= int(df_end.timestamp.to_numpy())]
    return df
