__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"


import numpy as np
import pandas as pd
from src.tools import pandas_from_topic, crop_df

HOVER_PWM = 1500


class FlightTimeSelector():
    # defines start and end time of flight using the value of actuator 1
    def __init__(self, ulog):
        # check the time when actuator 1 is larger than the minimum PWM required for hover
        act_df = pandas_from_topic(ulog, ["actuator_outputs"])
        act_df_crp = act_df[act_df.iloc[:, 2] > HOVER_PWM]

        # set start and end time of flight duration
        self.t_start = act_df_crp.iloc[1, 0]
        self.t_end = act_df_crp.iloc[(act_df_crp.shape[0]-1), 0]

        # print("Flight Time Selector Initialized")
        # print("start time: ", self.t_start)
        # print("end time:   ", self.t_end)

    def crop_to_fight_time(self, df):
        return crop_df(df, self.t_start, self.t_end)
