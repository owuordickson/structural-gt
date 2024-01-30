# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

# ONLY ADD MODULES
from .modules.graph_metrics import GraphMetrics
from .modules.graph_skeleton import GraphSkeleton
from .modules.graph_struct import GraphStruct

from .entrypoints import main_cli
from .entrypoints import main_gui


__version__ = "0.0.2"

__all__ = ['GraphMetrics', 'GraphStruct']  # packages available in 'from chaosgt import *'
