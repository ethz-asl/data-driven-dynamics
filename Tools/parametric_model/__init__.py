import os
from . import src
from . import tests

data_selection = os.getenv('data_selection')
if data_selection == "True":
    from . import visual_dataframe_selector
