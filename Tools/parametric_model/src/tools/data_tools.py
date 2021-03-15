
__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import pandas as pd
from pyulog import messages, info, core


def load_ulog(ulog_path):
    ulog = core.ULog(ulog_path)
    return ulog


def pandas_from_topic(ulog, topic_list):

    topics_df = pd.DataFrame()

    for topic in topic_list:
        try:
            topic_data = ulog.get_dataset(topic)
        except (KeyError, IndexError, ValueError) as error:
            print(type(error), topic, ":", error)

        curr_df = pd.DataFrame.from_dict(topic_data.data)
        if topics_df.empty:
            topics_df = curr_df
        else:
            topics_df = pd.concat([topics_df, curr_df], axis=1)

    return topics_df
