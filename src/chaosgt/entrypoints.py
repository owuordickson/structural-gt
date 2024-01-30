# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

"""
Entry points that allow users to execute GUI or Cli programs
"""


from .apps._gui_main import ChaosGUI
from .apps._cli_main import terminal_app


def main_gui():
    app = ChaosGUI()
    app.mainloop()


def main_cli():
    terminal_app()
