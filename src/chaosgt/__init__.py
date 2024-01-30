# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.
"""
A software package for performing Graph Theory and Chaos analysis on microscopic TEM images
"""

# MODULES
from .modules.graph_metrics import GraphMetrics
from .modules.graph_skeleton import GraphSkeleton
from .modules.graph_struct import GraphStruct

# APPs
from .entrypoints import main_cli
from .entrypoints import main_gui

# Project version
__version__ = "0.0.2"

# Packages available in 'from chaosgt import *'
__all__ = ['GraphMetrics', 'GraphSkeleton', 'GraphStruct']
