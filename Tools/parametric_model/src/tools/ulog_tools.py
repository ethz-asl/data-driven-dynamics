__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

import pandas as pd
import os

from pathlib import Path
from pyulog import core


def load_ulog(rel_ulog_path):
    proj_path = Path(os.getcwd()).parent.parent
    log_file_path = os.path.join(proj_path, rel_ulog_path)
    ulog = core.ULog(log_file_path)
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
