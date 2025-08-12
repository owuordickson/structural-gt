"""Exposes Python methods to be called by the GUI."""

import os
import cv2
import time
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from .checkbox_model import CheckBoxModel



class MainController(QObject):

    updateProgress = pyqtSignal(int, str)
    changeImageSignal = pyqtSignal()
    imageChangedSignal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_cv = None
        dummy_data = [
            {"id": 1, "text": "Apply Median Filter", "value": 0},
            {"id": 2, "text": "Apply Scharr Filter", "value": 1},
            {"id": 3, "text": "Swap Threshold", "value": 0}
        ]
        self.imgFilterModel = CheckBoxModel(dummy_data)

    def wait(self):
        self.updateProgress.emit(0, "Processing your name...")
        time.sleep(5)
        self.process_image()
        self.updateProgress.emit(100, "We finished processing your name!")

    @pyqtSlot()
    def process_image(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        img_path = os.path.join(script_dir, 'rGO.jpeg')
        self.img_cv = cv2.imread(img_path)
        if self.img_cv is None:
            self.updateProgress.emit(-1, "Could not read image")
            return
        self.updateProgress.emit(100, "Image added!")
        self.changeImageSignal.emit()

    @pyqtSlot(result='QString')
    def get_pixmap(self):
        unique_id = np.random.randint(1, 1000)
        return "image://imageProvider/" + str(unique_id)

    @pyqtSlot()
    def apply_filter_changes(self):
        """Retrieve changes made by the user and apply to image/graph."""
        self.changeImageSignal.emit()

    @pyqtSlot('QString', result='QString')
    def process_name(self, name: str) -> str:
        """Process the given name and return a greeting message."""
        self.wait()
        return f"Hey {name}, your name has been processed successfully."
