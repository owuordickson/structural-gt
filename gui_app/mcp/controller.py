"""Exposes Python methods to be called by the GUI."""

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class MainController(QObject):

    showAlertSignal = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)

    @pyqtSlot('QString', result='QString')
    def process_name(self, name: str) -> str:
        """Process the given name and return a greeting message."""
        return f"Hey {name}, your name has been processed successfully."
