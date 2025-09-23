# SPDX-License-Identifier: GNU GPL v3

"""
A launcher for executing the application as a Window app or a Terminal app.
"""

from multiprocessing import freeze_support
from sgtlib.entrypoints import main_gui

if __name__ == "__main__":
    freeze_support()
    main_gui()
