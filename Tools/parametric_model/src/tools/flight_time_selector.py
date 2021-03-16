__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


import numpy as np
import pandas as pd
from src.tools import pandas_from_topic, crop_df


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

    def crop_to_fight_time(self, df):
        return crop_df(df, self.t_start, self.t_end)
