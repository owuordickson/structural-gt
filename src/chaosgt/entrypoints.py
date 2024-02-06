# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs
"""


try:
    from .apps.gui_main_tk import ChaosGUI
except ImportError or ModuleNotFoundError:
    print("Please install 'Python3-tk'")
from .apps.cli_main import terminal_app


def main_gui():
    try:
        app = ChaosGUI()
        app.mainloop()
    except NameError:
        raise ModuleNotFoundError("Please install 'Python3-tk'")


def main_cli():
    terminal_app()
