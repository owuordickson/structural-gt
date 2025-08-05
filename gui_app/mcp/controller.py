"""Exposes Python methods to be called by the GUI."""

from PySide6.QtCore import QObject, Slot


class MainController(QObject):

    def __init__(self):
        super().__init__()
        
    @Slot(str, result=str)
    def process_name(self, name: str) -> str:
        """Process the given name and return a greeting message."""
        print(f"Welcome {name}!")
        return f"Hey {name}, your name has been processed successfully."
