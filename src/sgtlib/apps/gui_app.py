# SPDX-License-Identifier: GNU GPL v3

"""
Pyside6 implementation of StructuralGT user interface.
"""

import os
import sys
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
# from PySide6.QtQuickControls2 import QQuickStyle

from .controllers.main_controller import MainController
from .models.image_provider import ImageProvider

class PySideApp(QObject):

    @staticmethod
    def _force_backend():
        # High DPI fixes
        # QApplication.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        # QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

        # Force OpenGL backend for QML rendering (instead of D3D11)
        os.environ["QSG_RHI_BACKEND"] = "opengl"

        # Ensure Qt finds the right plugins (like when bundled)
        # Adjust the path if PySide6 is installed elsewhere
        import PySide6
        qt_plugins = os.path.join(os.path.dirname(PySide6.__file__), "plugins")
        os.environ["QT_PLUGIN_PATH"] = qt_plugins

    def _initialize_models(self):
        """Initialize the models and providers used by the QML engine."""
        self._ui_engine.addImageProvider("imageProvider", self._image_provider)

        self._ui_engine.rootContext().setContextProperty("graphPropsModel", self._ctrl.graph_ctrl.graphPropsModel)
        self._ui_engine.rootContext().setContextProperty("graphComputeModel", self._ctrl.graph_ctrl.graphComputeModel)
        self._ui_engine.rootContext().setContextProperty("gteTreeModel", self._ctrl.graph_ctrl.gteTreeModel)
        self._ui_engine.rootContext().setContextProperty("gtcListModel", self._ctrl.graph_ctrl.gtcListModel)
        self._ui_engine.rootContext().setContextProperty("gtcScalingModel", self._ctrl.graph_ctrl.gtcScalingModel)
        self._ui_engine.rootContext().setContextProperty("exportGraphModel", self._ctrl.graph_ctrl.exportGraphModel)

        self._ui_engine.rootContext().setContextProperty("microscopyPropsModel", self._ctrl.img_ctrl.microscopyPropsModel)
        self._ui_engine.rootContext().setContextProperty("imagePropsModel", self._ctrl.img_ctrl.imagePropsModel)
        self._ui_engine.rootContext().setContextProperty("imgBatchModel", self._ctrl.img_ctrl.imgBatchModel)
        self._ui_engine.rootContext().setContextProperty("imgControlModel", self._ctrl.img_ctrl.imgControlModel)
        self._ui_engine.rootContext().setContextProperty("imgBinFilterModel", self._ctrl.img_ctrl.imgBinFilterModel)
        self._ui_engine.rootContext().setContextProperty("imgFilterModel", self._ctrl.img_ctrl.imgFilterModel)
        self._ui_engine.rootContext().setContextProperty("imgColorsModel", self._ctrl.img_ctrl.imgColorsModel)
        self._ui_engine.rootContext().setContextProperty("imgScaleOptionModel", self._ctrl.img_ctrl.imgScaleOptionModel)
        self._ui_engine.rootContext().setContextProperty("imgViewOptionModel", self._ctrl.img_ctrl.imgViewOptionModel)
        self._ui_engine.rootContext().setContextProperty("saveImgModel", self._ctrl.img_ctrl.saveImgModel)
        self._ui_engine.rootContext().setContextProperty("img3dGridModel", self._ctrl.img_ctrl.img3dGridModel)
        self._ui_engine.rootContext().setContextProperty("imgHistogramModel", self._ctrl.img_ctrl.imgHistogramModel)

        self._ui_engine.rootContext().setContextProperty("imgThumbnailModel", self._ctrl.proj_ctrl.imgThumbnailModel)
        self._ui_engine.rootContext().setContextProperty("aiSearchModel", self._ctrl.ai_ctrl.aiSearchModel)


    def _initialize_controllers(self):
        """Initialize the controllers used by the QML engine."""
        self._ui_engine.rootContext().setContextProperty("mainController", self._ctrl)
        self._ui_engine.rootContext().setContextProperty("aiController", self._ctrl.ai_ctrl)
        self._ui_engine.rootContext().setContextProperty("imageController", self._ctrl.img_ctrl)
        self._ui_engine.rootContext().setContextProperty("graphController", self._ctrl.graph_ctrl)
        self._ui_engine.rootContext().setContextProperty("projectController", self._ctrl.proj_ctrl)

    def __init__(self):
        super().__init__()
        # PySideApp._force_backend()
        self.app = QApplication(sys.argv)
        self._ui_engine = QQmlApplicationEngine()
        # Register Controller for Dynamic Updates
        self._ctrl = MainController(qml_app=self.app)
        # Register Image Provider
        self._image_provider = ImageProvider(self._ctrl)
        self._qml_file = 'qml/MainWindow.qml'

        # Set Models in QML Context
        self._initialize_models()
        self._initialize_controllers()

        # Cleanup when the app is closing
        self.app.aboutToQuit.connect(self._ctrl.cleanup_workers)

        # Load UI
        # Get the directory of the current script
        qml_dir = os.path.dirname(os.path.abspath(__file__))
        qml_path = os.path.join(qml_dir, self._qml_file)

        # Set Theme for the entire application UI ('Basic', 'Fusion', 'Imagine', 'Material', 'Universal')
        # QQuickStyle.setStyle("Basic")

        # Load the QML file and display it
        self._ui_engine.load(qml_path)
        if not self._ui_engine.rootObjects():
            sys.exit(-1)

    @classmethod
    def start(cls) -> None:
        """
        Initialize and run the PySide GUI application.
        """
        gui_app = cls()
        sys.exit(gui_app.app.exec())
