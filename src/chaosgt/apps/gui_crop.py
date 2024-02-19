from PyQt6.QtWidgets import QDialog
from PyQt6.QtCore import QRect, QSize, QSignalBlocker, pyqtSlot
from .ui_crop.gui_qcrop import Ui_QCrop

MARGIN_H = 48
MARGIN_V = 120


class QCrop(QDialog):

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)

        self._ui = Ui_QCrop()
        self._ui.setupUi(self)

        self.setWindowTitle('QCrop - v{}'.format('0.0.1'))

        self.image = pixmap
        self._original = QRect(0, 0, self.image.width(), self.image.height())
        self._ui.selector.crop = QRect(0, 0, self.image.width(), self.image.height())
        self._ui.selector.setPixmap(self.image)

        self._ui.spinBoxX.setMaximum(self._original.width()-1)
        self._ui.spinBoxY.setMaximum(self._original.height()-1)
        self._ui.spinBoxW.setMaximum(self._original.width())
        self._ui.spinBoxH.setMaximum(self._original.height())
        self.update_crop_values()

        self.resize(self._original.width() + MARGIN_H, self._original.height() + MARGIN_V)

    def update_crop_area(self):
        values = self.crop_values()
        if self._ui.selector.crop != values:
            self._ui.selector.crop = values
            self._ui.selector.update()

    def crop_values(self):
        return QRect(
            self._ui.spinBoxX.value(),
            self._ui.spinBoxY.value(),
            self._ui.spinBoxW.value(),
            self._ui.spinBoxH.value()
        )

    def update_crop_values(self):
        self._ui.spinBoxX.blockSignals(True)
        self._ui.spinBoxX.setValue(self._ui.selector.crop.x())
        self._ui.spinBoxX.blockSignals(False)
        self._ui.spinBoxY.blockSignals(True)
        self._ui.spinBoxY.setValue(self._ui.selector.crop.y())
        self._ui.spinBoxY.blockSignals(False)
        self._ui.spinBoxW.blockSignals(True)
        self._ui.spinBoxW.setValue(self._ui.selector.crop.width())
        self._ui.spinBoxW.blockSignals(False)
        self._ui.spinBoxH.blockSignals(True)
        self._ui.spinBoxH.setValue(self._ui.selector.crop.height())
        self._ui.spinBoxH.blockSignals(False)

    @pyqtSlot()
    def reset_crop_values(self):
        self._ui.spinBoxX.setValue(0)
        self._ui.spinBoxY.setValue(0)
        self._ui.spinBoxW.setValue(self._original.width())
        self._ui.spinBoxH.setValue(self._original.height())

    @pyqtSlot()
    def accept(self):
        if self._ui.selector.crop != self._original:
            self.image = self.image.copy(self._ui.selector.crop)
            print("Yes")
            super().accept()
        else:
            print("No")
            super().reject()
