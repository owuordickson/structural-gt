# SPDX-License-Identifier: GNU GPL v3

"""
Terminal interface implementations
"""

import time
import os
import logging

from ..utils.sgt_utils import get_num_cores
from ..imaging.image_processor import ImageProcessor, FiberNetworkBuilder
from ..compute.graph_analyzer import GraphAnalyzer


def terminal_app():
    """
    Initializes and executes StructuralGT functions.
    :return:
    """
    configs = load_project_configs()
    alg = configs.algChoice
