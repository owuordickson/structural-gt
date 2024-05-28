"""
**StructuralGTb**

A software package for performing Graph Theory on microscopic TEM images. This software is a \
modified version of StructuralGT by Drew A. Vecchio: https://github.com/drewvecchio/StructuralGT.

    Copyright (C) 2024, the Regents of the University of Michigan.

        This program is free software: you can redistribute it and/or modify \
it under the terms of the GNU General Public License as published by \
the Free Software Foundation, either version 3 of the License, or \
(at your option) any later version. This program is distributed in \
the hope that it will be useful, but WITHOUT ANY WARRANTY; without \
even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  \
See the GNU General Public License for more details. You should have received a copy \
of the GNU General Public License along with this program.  If not, see <https://www.gnu.org/licenses/>.

        Development Lead: Dickson Owuor

        Contributors: Nicholas A. Kotov

        Contact email: owuordickson@gmail.com
"""

# MODULES
from .SGT.graph_metrics import GraphMetrics
from .SGT.graph_skeleton import GraphSkeleton
from .SGT.graph_converter import GraphConverter

# APPs
from .entrypoints import main_cli
from .entrypoints import main_gui

# Project version
__version__ = "1.2.7"
__author__ = "Dickson Owuor"
__credits__ = "The Regents of the University of Michigan"


# Packages available in 'from StructuralGTb import *'
__all__ = ['GraphMetrics', 'GraphSkeleton', 'GraphConverter', 'main_cli', 'main_gui']
