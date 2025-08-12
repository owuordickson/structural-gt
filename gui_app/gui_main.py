"""
Pyside6 implementation of app user interface.
"""

import os
import sys
from PyQt6.QtCore import QObject
from PyQt6.QtWidgets import QApplication
from PyQt6.QtQml import QQmlApplicationEngine
from mcp.controller import MainController
from mcp.image_provider import ImageProvider

class MainApp(QObject):
    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self.ui_engine = QQmlApplicationEngine()

        # Register Controller for Dynamic Updates
        controller = MainController()
        # Register Image Provider
        self.image_provider = ImageProvider(controller)

        # Set Models/Controllers in QML Context
        self.ui_engine.rootContext().setContextProperty("mainController", controller)
        self.ui_engine.rootContext().setContextProperty("imgFilterModel", controller.imgFilterModel)
        self.ui_engine.addImageProvider("imageProvider", self.image_provider)

        # Load UI
        # Get the directory of the current script
        qml_dir = os.path.dirname(os.path.abspath(__file__))
        qml_name = 'qml/MainWindow.qml'
        qml_path = os.path.join(qml_dir, qml_name)
        self.ui_engine.load(qml_path)
        if not self.ui_engine.rootObjects():
            print("Could not start GUI")
            sys.exit(-1)


if __name__ == "__main__":
    # Start GUI app
    py_app = MainApp()
    sys.exit(py_app.app.exec())
