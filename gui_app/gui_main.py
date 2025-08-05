"""
Pyside6 implementation of app user interface.
"""

import os
import sys
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from mcp.controller import MainController

class MainWindow(QObject):
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.ui_engine = QQmlApplicationEngine()

        # Register Controller for Dynamic Updates
        controller = MainController()

        # Set Models/Controllers in QML Context
        self.ui_engine.rootContext().setContextProperty("mainController", controller)

        # Load UI
        # Get the directory of the current script
        qml_dir = os.path.dirname(os.path.abspath(__file__))
        qml_name = 'qml/MainWindow.qml'
        qml_path = os.path.join(qml_dir, qml_name)
        self.ui_engine.load(qml_path)
        if not self.ui_engine.rootObjects():
            sys.exit(-1)


def pyside_app() -> None:
    """
    Initialize and run the PySide GUI application.
    Returns:

    """
    main_window = MainWindow()
    sys.exit(main_window.app.exec())


if __name__ == "__main__":
    # Start GUI app
    pyside_app()
