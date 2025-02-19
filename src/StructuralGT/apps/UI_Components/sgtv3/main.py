import sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject

from gui_controller import MainController
from gui_image_provider import ImageProvider


# Assuming TreeModel and TableModel are properly implemented
class MainWindow(QObject):
    def __init__(self):
        super().__init__()
        self.app = QGuiApplication(sys.argv)
        self.ui_engine = QQmlApplicationEngine()

        # Register Controller for Dynamic Updates
        controller = MainController()
        # Register Image Provider
        self.image_provider = ImageProvider(controller)

        # Load Model Data
        controller.load_data()

        # Load Default Configs
        # controller.load_default_configs()

        # Set Models in QML Context
        self.ui_engine.rootContext().setContextProperty("graphTreeModel", controller.graphTreeModel)
        self.ui_engine.rootContext().setContextProperty("graphPropsTableModel", controller.graphPropsTableModel)
        self.ui_engine.rootContext().setContextProperty("imgPropsTableModel", controller.imgPropsTableModel)
        self.ui_engine.rootContext().setContextProperty("imgListTableModel", controller.imgListTableModel)
        self.ui_engine.rootContext().setContextProperty("mainController", controller)
        self.ui_engine.addImageProvider("imageProvider", self.image_provider)

        # Load UI
        self.ui_engine.load("MainWindow.qml")
        if not self.ui_engine.rootObjects():
            sys.exit(-1)


if __name__ == "__main__":
    main_window = MainWindow()
    sys.exit(main_window.app.exec())
