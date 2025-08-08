# SPDX-License-Identifier: GNU GPL v3

"""
QThread Worker class for running tasks in the background.
"""

from PySide6.QtCore import QThread


class QThreadWorker(QThread):
    def __init__(self, func, args, parent=None):
        super().__init__(parent)
        self.func = func  # Store function reference
        self.args = args  # Store arguments

    def run(self):
        if self.func:
            self.func(*self.args)  # Call function with arguments

