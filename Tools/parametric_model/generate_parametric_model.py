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

__author__ = "Manuel Yves Galliker, Jaeyoung Lim"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

import os
import src.models as models
from src.tools import DataHandler
import argparse
import pandas as pd


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        os.environ['data_selection'] = "True"
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        os.environ['data_selection'] = "False"
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def start_model_estimation(config, log_path, data_selection="none", plot=False):
    print("Visual Data selection enabled: ", data_selection)

    # Flag for enabling automatic data selection.

    data_handler = DataHandler(config)
    data_handler.loadLogs(log_path)

    data_df = data_handler.get_dataframes()

    model_class = data_handler.config.model_class
    try:
        # This will call the model constructor directly from the model_class
        # in the yaml config (self-describing)
        # i.e if model_name is MultiRotorModel then it will call that __init__()
        model = getattr(models, model_class)(config)
    except AttributeError:
        error_str = "Model '{0}' not found, is it added to models "\
                    "directory and models/__init__.py?".format(model_class)
        raise AttributeError(error_str)

    visual_dataframe_selector_config_dict = {
    "x_axis_col": "timestamp",
    "sub_plt1_data": ["q0", "q1", "q2", "q3"],
    "sub_plt2_data": ["u0", "u1", "u2", "u3"],
    "sub_plt3_data": []
    }

    if data_handler.estimate_forces == True:
        visual_dataframe_selector_config_dict["sub_plt3_data"].append("fisher_information_force")

    if data_handler.estimate_moments == True:
        visual_dataframe_selector_config_dict["sub_plt3_data"].append("fisher_information_rot")

    model.load_dataframes(data_df)
    model.prepare_regression_matrices()
    model.compute_fisher_information()

    # Interactive data selection
    if data_selection=="interactive":
        from visual_dataframe_selector.data_selector import select_visual_data
        model.data_df = select_visual_data(model.data_df,visual_dataframe_selector_config_dict)
        model.n_samples = model.data_df.shape[0]
    # Automatic data selection (WIP)
    elif data_selection=="auto":
        from active_dataframe_selector.data_selector import ActiveDataSelector
        # The goal is to identify automatically the most relevant parts of a log.
        # Currently the draft is designed to choose the most informative 10% of the logs with regards to
        # force and moment parameters. This threshold is currently not validated at all and the percentage
        # can vary drastically from log to log. 

        data_selector = ActiveDataSelector(model.data_df)
        model.data_df = data_selector.select_dataframes(10)
        model.n_samples = model.data_df.shape[0]

    model.estimate_model()

    if plot:
        model.compute_residuals()
        model.plot_model_predicitons()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Estimate dynamics model from flight log.')
    parser.add_argument('log_path', metavar='log_path', type=str,
                        help='The path of the log to process relative to the project directory.')
    parser.add_argument('--data_selection', metavar='data_selection', type=str, default="none",
                        help='Data selection scheme none | interactive | auto (Beta)')
    parser.add_argument('--config', metavar='config', type=str, default='configs/quadrotor_model.yaml',
                        help='Configuration file path for pipeline configurations')
    parser.add_argument('--plot', metavar='plot', type=str2bool, default='True',
                        help='Show plots after fit.')
    arg_list = parser.parse_args()
    start_model_estimation(**vars(arg_list))
