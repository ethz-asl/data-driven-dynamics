__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import numpy as np
import pandas as pd
from src.tools import pandas_from_topic


def interpolate_dataframes(df_list, t_start, t_end):
    # create a single dataframe by downsampling to the frame with the lowest frequency
    min_length = None
    for df in df_list:
        df = crop_df(df, t_start, t_end)
        if (min_length is None):
            min_length = df.shape[0]
            shortest_df = df
        elif(min_length > df.shape[0]):
            min_length = df.shape[0]
            shortest_df = df
    for df in df_list:
        if (not df.equals(shortest_df)):
            new_df = pd.DataFrame(
                np.zeros((shortest_df.shape[0], df.shape[1])), columns=df.columns)
            print(new_df)
            # for i in range(min_length):
            #     fff = 0


def crop_df(df, t_start, t_end):
    df = df[df.iloc[:, 0] >= t_start]
    df = df[df.iloc[:, 0] <= t_end]
    return df


class FlightTimeSelector():
    # defines start and end time of flight using the value of actuator 1
    def __init__(self, ulog):
        # check the time when actuator 1 is larger than default arming value of 900
        act_df = pandas_from_topic(ulog, ["actuator_outputs"])
        act_df_crp = act_df[act_df.iloc[:, 2] > 900.0]

        # set start and end time of flight duration
        self.t_start = act_df_crp.iloc[1, 0]
        self.t_end = act_df_crp.iloc[(act_df_crp.shape[0]-1), 0]

        print(self.t_start, self.t_end)

    def crop_to_fight_data(self, df):
        return crop_df(df, self.t_start, self.t_end)
