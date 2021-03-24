__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
from src.tools import pandas_from_topic

HOVER_PWM = 1500


def compute_flight_time(ulog):
    act_df = pandas_from_topic(ulog, ["actuator_outputs"])
    act_df_crp = act_df[act_df.iloc[:, 2] > HOVER_PWM]

    # set start and end time of flight duration
    t_start = act_df_crp.iloc[1, 0]
    t_end = act_df_crp.iloc[(act_df_crp.shape[0]-1), 0]
    flight_time = {"t_start": t_start, "t_end": t_end}
    return flight_time


def resample_dataframes(df_list, t_start, t_end, f_des=100.0):
    """create a single dataframe by resampling all dataframes to f_des [Hz]

    Inputs:     df_list : List of ulog topic dataframes to resample
                t_start : Start time in us
                t_end   : End time in us
                f_des   : Desired frequency of resampled data   
    """

    # compute desired Period in us to be persistent with ulog timestamps
    assert f_des > 0, 'Desired frequency must be greater than 0'
    T_des = 1000000.0/f_des

    n_samples = int((t_end-t_start)/T_des)
    res_df = pd.DataFrame()
    for df in df_list:
        df = crop_df(df, t_start, t_end)
        new_df = pd.DataFrame()
        new_t = np.arange(t_start, t_end, T_des)
        for col in df:
            new_df[col] = np.interp(new_t, df.timestamp, df[col])

        res_df = pd.concat([res_df, new_df], axis=1)

    return res_df.T.drop_duplicates().T


def crop_df(df, t_start, t_end):
    """ crop df to contain 1 elemnt before t_start and one after t_end.
    This way it is easy to interpolate the data between start and end time. """
    df_start = df[df.timestamp <= t_start].iloc[[-1]]
    df_end = df[df.timestamp >= t_end].iloc[[0]]

    df = df[df.timestamp >= int(df_start.timestamp.to_numpy())]
    df = df[df.timestamp <= int(df_end.timestamp.to_numpy())]
    return df
