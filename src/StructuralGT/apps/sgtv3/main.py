import sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject

from gui_controller import MainController
from gui_image_provider import ImageProvider


class MainWindow(QObject):
    def __init__(self):
        super().__init__()
        self.app = QGuiApplication(sys.argv)
        self.ui_engine = QQmlApplicationEngine()

        # Register Controller for Dynamic Updates
        controller = MainController()
        # Register Image Provider
        self.image_provider = ImageProvider(controller)

        # Test Image
        # img_path = "../../../../../datasets/InVitroBioFilm.png"
        # controller.imageChangedSignal.emit(0, img_path)

        # Set Models in QML Context
        self.ui_engine.rootContext().setContextProperty("graphPropsTableModel", controller.graphPropsTableModel)
        self.ui_engine.rootContext().setContextProperty("imgPropsTableModel", controller.imgPropsTableModel)
        self.ui_engine.rootContext().setContextProperty("imgListTableModel", controller.imgListTableModel)

        self.ui_engine.rootContext().setContextProperty("gteTreeModel", controller.gteTreeModel)
        self.ui_engine.rootContext().setContextProperty("gtcListModel", controller.gtcListModel)
        self.ui_engine.rootContext().setContextProperty("imgBinFilterModel", controller.imgBinFilterModel)
        self.ui_engine.rootContext().setContextProperty("imgFilterModel", controller.imgFilterModel)
        self.ui_engine.rootContext().setContextProperty("imgControlModel", controller.imgControlModel)
        self.ui_engine.rootContext().setContextProperty("microscopyPropsModel", controller.microscopyPropsModel)
        self.ui_engine.rootContext().setContextProperty("mainController", controller)
        self.ui_engine.addImageProvider("imageProvider", self.image_provider)

        # Load UI
        self.ui_engine.load("MainWindow.qml")
        if not self.ui_engine.rootObjects():
            sys.exit(-1)


if __name__ == "__main__":
    main_window = MainWindow()
    sys.exit(main_window.app.exec())
