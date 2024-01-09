"""
**ChaosGT**

A python package for chaos engineering in nano-structures.

Copyright (C) 2024, The University of Michigan.

This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

Development Lead: Dickson Owuor
Contributors: Nicholas A. Kotov
Contact email: owuordickson@gmail.com
"""

__author__ = "Dickson Owuor"
__credits__ = "Chemical Engineering Department, University of Michigan"


from chaosgt.entrypoints import main_gui

if __name__ == "__main__":
    main_gui()
