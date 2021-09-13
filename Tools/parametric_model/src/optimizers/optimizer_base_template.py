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

__author__ = "Manuel Yves Galliker"
__maintainer__ = "Manuel Yves Galliker"
__license__ = "BSD 3"

from abc import ABC, abstractmethod
from typing import Dict, List


class ParametersNotEstimatedError(Exception):
    def __init__(self):
        self.message = "Optimization parameters are not yet estimated." + \
            "Did you call estimate_parameters() method?"
        super().__init__(self.message)


class OptimizerBaseTemplate(ABC):

    """ 
    This is an abstract class that is used as the base 
    template to all other optimizer classes by overwriting the 
    abstract methods. 
    """

    def __init__(self, optimizer_config, param_name_list):
        self.estimation_completed = False
        self.parametersNotEstimatedError = ParametersNotEstimatedError()
        self.config = optimizer_config
        self.param_name_list = param_name_list

    def check_estimation_completed(self):
        if self.estimation_completed:
            return
        else:
            raise self.parametersNotEstimatedError

    @abstractmethod
    def estimate_parameters(self) -> None:
        pass

    @abstractmethod
    def set_optimal_coefficients(self) -> None:
        pass

    @abstractmethod
    def get_optimization_parameters(self) -> List:
        pass

    @abstractmethod
    def predict(self) -> List:
        pass

    @abstractmethod
    def compute_optimization_metrics(self) -> Dict:
        pass
