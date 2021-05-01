#!/usr/bin/env bash

export PYTHONPATH=$(dirname "$0")
pytest Tools/parametric_model/tests/test_dynamics_model.py