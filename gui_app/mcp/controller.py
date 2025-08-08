"""Exposes Python methods to be called by the GUI."""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
import time


class MainController(QObject):

    updateProgress = pyqtSignal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    def wait(self):
        self.updateProgress.emit(0, "Processing your name...")
        time.sleep(5)
        self.updateProgress.emit(100, "We finished processing your name!")

    @pyqtSlot('QString', result='QString')
    def process_name(self, name: str) -> str:
        """Process the given name and return a greeting message."""
        self.wait()
        return f"Hey {name}, your name has been processed successfully."
