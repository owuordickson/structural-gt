# SPDX-License-Identifier: GNU GPL v3

"""
Pyside6 implementation of StructuralGT user interface.
"""

import os
import sys
from PySide6.QtCore import QObject
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine

from .gui_mcw.controller import MainController
from .gui_mcw.image_provider import ImageProvider


class PySideApp(QObject):

    def __init__(self):
        super().__init__()
        self.app = QApplication(sys.argv)
        self._ui_engine = QQmlApplicationEngine()
        # Register Controller for Dynamic Updates
        self._controller = MainController(qml_app=self.app)
        # Register Image Provider
        self._image_provider = ImageProvider(self._controller)
        self._qml_file = 'sgt_qml/MainWindow.qml'

        # Set Models in QML Context
        self._ui_engine.rootContext().setContextProperty("imgThumbnailModel", self._controller.imgThumbnailModel)
        self._ui_engine.rootContext().setContextProperty("imagePropsModel", self._controller.imagePropsModel)
        self._ui_engine.rootContext().setContextProperty("graphPropsModel", self._controller.graphPropsModel)
        self._ui_engine.rootContext().setContextProperty("graphComputeModel", self._controller.graphComputeModel)
        self._ui_engine.rootContext().setContextProperty("microscopyPropsModel", self._controller.microscopyPropsModel)

        self._ui_engine.rootContext().setContextProperty("gteTreeModel", self._controller.gteTreeModel)
        self._ui_engine.rootContext().setContextProperty("gtcListModel", self._controller.gtcListModel)
        self._ui_engine.rootContext().setContextProperty("gtcScalingModel", self._controller.gtcScalingModel)
        self._ui_engine.rootContext().setContextProperty("exportGraphModel", self._controller.exportGraphModel)
        self._ui_engine.rootContext().setContextProperty("imgBatchModel", self._controller.imgBatchModel)
        self._ui_engine.rootContext().setContextProperty("imgControlModel", self._controller.imgControlModel)
        self._ui_engine.rootContext().setContextProperty("imgBinFilterModel", self._controller.imgBinFilterModel)
        self._ui_engine.rootContext().setContextProperty("imgFilterModel", self._controller.imgFilterModel)
        self._ui_engine.rootContext().setContextProperty("imgScaleOptionModel", self._controller.imgScaleOptionModel)
        self._ui_engine.rootContext().setContextProperty("imgViewOptionModel", self._controller.imgViewOptionModel)
        self._ui_engine.rootContext().setContextProperty("saveImgModel", self._controller.saveImgModel)
        self._ui_engine.rootContext().setContextProperty("img3dGridModel", self._controller.img3dGridModel)
        self._ui_engine.rootContext().setContextProperty("imgHistogramModel", self._controller.imgHistogramModel)
        self._ui_engine.rootContext().setContextProperty("mainController", self._controller)
        self._ui_engine.addImageProvider("imageProvider", self._image_provider)

        # Load UI
        # Get the directory of the current script
        qml_dir = os.path.dirname(os.path.abspath(__file__))
        qml_path = os.path.join(qml_dir, self._qml_file)
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
