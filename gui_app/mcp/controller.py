"""Exposes Python methods to be called by the GUI."""

import os
import cv2
import time
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot



class MainController(QObject):

    updateProgress = pyqtSignal(int, str)
    changeImageSignal = pyqtSignal()
    imageChangedSignal = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_cv = None

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
            print("Could not read image")
            return
        self.changeImageSignal.emit()

    @pyqtSlot(result='QString')
    def get_pixmap(self):
        unique_id = np.random.randint(1, 1000)
        return "image://imageProvider/" + str(unique_id)

    @pyqtSlot('QString', result='QString')
    def process_name(self, name: str) -> str:
        """Process the given name and return a greeting message."""
        self.wait()
        return f"Hey {name}, your name has been processed successfully."
