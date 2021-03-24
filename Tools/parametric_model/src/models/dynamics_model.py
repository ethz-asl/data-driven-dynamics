__author__ = "Manuel Galliker"
__maintainer__ = "Manuel Galliker"
__license__ = "BSD 3"

""" The model class contains properties shared between all models and shgall simplyfy automated checks and the later 
export to a sitl gazebo model by providing a unified interface for all models. """

from ..tools import load_ulog


class DynamicsModel():
    def __init__(self, rel_ulog_path, req_topics_list):
        assert type(req_topics_list) is list, 'topic_list input must be a list'
        self.ulog = load_ulog(rel_ulog_path)
        self.req_topics = req_topics_list

    def check_ulog_for_req_topics(self):
        for topic in self.req_topics:
            try:
                topic_data = ulog.get_dataset(topic)
            except:
                return False
        return True
