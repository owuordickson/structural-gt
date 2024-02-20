"""

"""

import logging as log
from PyQt6.QtGui import QPainter, QBrush, QColor, QCursor, QPalette
from PyQt6.QtCore import QCoreApplication, QRect, QSize, QPoint, QMargins, QMetaObject, pyqtSignal, pyqtSlot, Qt
from PyQt6.QtWidgets import (QWidget, QLabel, QRubberBand, QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                             QSpinBox, QSpacerItem, QSizePolicy, QPushButton, QFrame, QScrollArea, QDialogButtonBox)


MARGIN_H = 48
MARGIN_V = 120


class QCrop(QDialog):

    def __init__(self, pixmap, parent=None):
        super().__init__(parent)

        self._ui = BaseUiQCrop(self)
        # self._ui.__createWidgets(self)

        self.setWindowTitle('QCrop - v{}'.format('0.0.2 sgt'))

        self.image = pixmap
        self._original = QRect(0, 0, self.image.width(), self.image.height())
        self._ui.selector.crop = QRect(0, 0, self.image.width(), self.image.height())
        self._ui.selector.setPixmap(self.image)

        self._ui.spinBoxX.setMaximum(self._original.width() - 1)
        self._ui.spinBoxY.setMaximum(self._original.height() - 1)
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
            super().accept()
        else:
            super().reject()


class BaseUiQCrop(object):

    def __init__(self, q_crop):
        q_crop.setObjectName("QCrop")
        q_crop.resize(664, 576)
        q_crop.setSizeGripEnabled(True)
        self.verticalLayout = QVBoxLayout(q_crop)
        self.verticalLayout.setSpacing(6)
        self.verticalLayout.setObjectName("verticalLayout")
        self.toolbar = QWidget(parent=q_crop)
        self.toolbar.setMaximumSize(QSize(16777215, 16777215))
        self.toolbar.setObjectName("toolbar")
        self.horizontalLayout = QHBoxLayout(self.toolbar)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.labelX = QLabel(parent=self.toolbar)
        self.labelX.setObjectName("labelX")
        self.horizontalLayout.addWidget(self.labelX)
        self.spinBoxX = QSpinBox(parent=self.toolbar)
        self.spinBoxX.setMinimumSize(QSize(72, 0))
        self.spinBoxX.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing |
            Qt.AlignmentFlag.AlignVCenter)
        self.spinBoxX.setObjectName("spinBoxX")
        self.horizontalLayout.addWidget(self.spinBoxX)
        self.labelY = QLabel(parent=self.toolbar)
        self.labelY.setObjectName("labelY")
        self.horizontalLayout.addWidget(self.labelY)
        self.spinBoxY = QSpinBox(parent=self.toolbar)
        self.spinBoxY.setMinimumSize(QSize(72, 0))
        self.spinBoxY.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing |
            Qt.AlignmentFlag.AlignVCenter)
        self.spinBoxY.setObjectName("spinBoxY")
        self.horizontalLayout.addWidget(self.spinBoxY)
        self.labelW = QLabel(parent=self.toolbar)
        self.labelW.setObjectName("labelW")
        self.horizontalLayout.addWidget(self.labelW)
        self.spinBoxW = QSpinBox(parent=self.toolbar)
        self.spinBoxW.setMinimumSize(QSize(72, 0))
        self.spinBoxW.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing |
            Qt.AlignmentFlag.AlignVCenter)
        self.spinBoxW.setObjectName("spinBoxW")
        self.horizontalLayout.addWidget(self.spinBoxW)
        self.labelH = QLabel(parent=self.toolbar)
        self.labelH.setObjectName("labelH")
        self.horizontalLayout.addWidget(self.labelH)
        self.spinBoxH = QSpinBox(parent=self.toolbar)
        self.spinBoxH.setMinimumSize(QSize(72, 0))
        self.spinBoxH.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTrailing |
            Qt.AlignmentFlag.AlignVCenter)
        self.spinBoxH.setObjectName("spinBoxH")
        self.horizontalLayout.addWidget(self.spinBoxH)
        spacer_item = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding,
                                  QSizePolicy.Policy.Minimum)
        self.horizontalLayout.addItem(spacer_item)
        self.pushButton = QPushButton(parent=self.toolbar)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout.addWidget(self.pushButton)
        self.verticalLayout.addWidget(self.toolbar)
        self.line = QFrame(parent=q_crop)
        self.line.setFrameShape(QFrame.Shape.HLine)
        self.line.setFrameShadow(QFrame.Shadow.Sunken)
        self.line.setObjectName("line")
        self.verticalLayout.addWidget(self.line)
        self.scrollArea = QScrollArea(parent=q_crop)
        self.scrollArea.viewport().setProperty("cursor", QCursor(Qt.CursorShape.CrossCursor))
        self.scrollArea.setAutoFillBackground(True)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.workspace = BaseWorkspace()
        self.workspace.setGeometry(QRect(0, 0, 638, 478))
        palette = QPalette()
        brush = QBrush(QColor(255, 255, 255))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Base, brush)
        brush = QBrush(QColor(51, 51, 51))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Active, QPalette.ColorRole.Window, brush)
        brush = QBrush(QColor(255, 255, 255))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Base, brush)
        brush = QBrush(QColor(51, 51, 51))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Window, brush)
        brush = QBrush(QColor(51, 51, 51))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, brush)
        brush = QBrush(QColor(51, 51, 51))
        brush.setStyle(Qt.BrushStyle.SolidPattern)
        palette.setBrush(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Window, brush)
        self.workspace.setPalette(palette)
        self.workspace.setObjectName("workspace")
        self.gridLayout = QGridLayout(self.workspace)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setSpacing(0)
        self.gridLayout.setObjectName("gridLayout")
        spacer_item1 = QSpacerItem(20, 224, QSizePolicy.Policy.Minimum,
                                   QSizePolicy.Policy.Expanding)
        self.gridLayout.addItem(spacer_item1, 2, 2, 1, 1)
        spacer_item2 = QSpacerItem(304, 20, QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacer_item2, 1, 3, 1, 1)
        self.frame = QFrame(parent=self.workspace)
        self.frame.setFrameShape(QFrame.Shape.Box)
        self.frame.setFrameShadow(QFrame.Shadow.Plain)
        self.frame.setObjectName("frame")
        self.verticalLayout_2 = QVBoxLayout(self.frame)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.selector = BaseAreaSelector(parent=self.frame)
        self.selector.setFrameShape(QFrame.Shape.NoFrame)
        self.selector.setText("")
        self.selector.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.selector.setObjectName("selector")
        self.verticalLayout_2.addWidget(self.selector)
        self.gridLayout.addWidget(self.frame, 1, 2, 1, 1)
        spacer_item3 = QSpacerItem(305, 20, QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Minimum)
        self.gridLayout.addItem(spacer_item3, 1, 0, 1, 1)
        spacer_item4 = QSpacerItem(20, 225, QSizePolicy.Policy.Minimum,
                                   QSizePolicy.Policy.Expanding)
        self.gridLayout.addItem(spacer_item4, 0, 2, 1, 1)
        self.scrollArea.setWidget(self.workspace)
        self.verticalLayout.addWidget(self.scrollArea)
        self.buttonBox = QDialogButtonBox(parent=q_crop)
        self.buttonBox.setOrientation(Qt.Orientation.Horizontal)
        self.buttonBox.setStandardButtons(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.__re_translate_ui(q_crop)
        self.buttonBox.accepted.connect(q_crop.accept)  # type: ignore
        self.buttonBox.rejected.connect(q_crop.reject)  # type: ignore
        self.selector.area_changed.connect(q_crop.update_crop_values)  # type: ignore
        self.spinBoxX.valueChanged['int'].connect(q_crop.update_crop_area)  # type: ignore
        self.spinBoxY.valueChanged['int'].connect(q_crop.update_crop_area)  # type: ignore
        self.spinBoxW.valueChanged['int'].connect(q_crop.update_crop_area)  # type: ignore
        self.spinBoxH.valueChanged['int'].connect(q_crop.update_crop_area)  # type: ignore
        self.pushButton.clicked.connect(q_crop.reset_crop_values)  # type: ignore
        QMetaObject.connectSlotsByName(q_crop)

    def __re_translate_ui(self, q_crop):
        _translate = QCoreApplication.translate
        q_crop.setWindowTitle(_translate("QCrop", "QCrop"))
        self.labelX.setText(_translate("QCrop", "X:"))
        self.spinBoxX.setSuffix(_translate("QCrop", "px"))
        self.labelY.setText(_translate("QCrop", "Y:"))
        self.spinBoxY.setSuffix(_translate("QCrop", "px"))
        self.labelW.setText(_translate("QCrop", "Width:"))
        self.spinBoxW.setSuffix(_translate("QCrop", "px"))
        self.labelH.setText(_translate("QCrop", "Height:"))
        self.spinBoxH.setSuffix(_translate("QCrop", "px"))
        self.pushButton.setText(_translate("QCrop", "Reset"))


class BaseAreaSelector(QLabel):
    area_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.crop = QRect(0, 0, 0, 0)
        # parent is the QFrame, its parent is the BaseWorkspace
        parent.parent().selector = self

    def paintEvent(self, event):
        super().paintEvent(event)

        qp = QPainter(self)
        qp.setBrush(QBrush(QColor(0, 0, 0, 200)))
        qp.setPen(Qt.PenStyle.NoPen)

        for r in BaseAreaSelector.exclude(self.geometry().translated(-1, -1), self.crop):
            qp.drawRect(r)

    @staticmethod
    def exclude(outer, inner):
        """
        :param outer: the external rectangle as a QRect object
        :param inner: the inner rectangle as a QRect object
        :return: a tuple of four QRect objects, representing the frame of the inner rect
            +----+--------------+--------+
            | 11 | 222222222222 | 444444 |
            | 11 +--------------+ 444444 |
            | 11 |              | 444444 |
            | 11 |              | 444444 |
            | 11 |              | 444444 |
            | 11 +--------------+ 444444 |
            | 11 | 333333333333 | 444444 |
            | 11 | 333333333333 | 444444 |
            +----+--------------+--------+
        """
        # just in case the inner rect goes beyond the outer limits
        log.debug('OUTER:')
        log.debug(outer)
        log.debug('INNER:')
        log.debug(inner)
        inner = outer.intersected(inner)
        areas = (
            QRect(outer.left(), outer.top(), inner.left(), outer.height()),
            QRect(inner.left(), outer.top(), inner.width(), inner.top()),
            QRect(inner.left(), inner.bottom() + 1, inner.width(), outer.height() - inner.bottom() - 1),
            QRect(inner.right() + 1, outer.top(), outer.width() - inner.right() + 1, outer.height())
        )
        log.debug('AREAS:')
        log.debug(areas)
        return areas


class BaseWorkspace(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._rubberband = None
        self._origin = None
        self.selector = None

        p = QPalette()
        p.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))  # Set color to black
        self.setPalette(p)

    def mousePressEvent(self, event):
        if not self._rubberband:
            self._rubberband = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._origin = event.pos()
        self.selector.crop = QRect(QPoint(0, 0), self.selector.geometry().size())
        self.repaint()
        self._rubberband.setGeometry(QRect(self._origin, QSize()))
        self._rubberband.show()

    def mouseMoveEvent(self, event):
        selection = QRect(self._origin, event.pos()).normalized()
        self._rubberband.setGeometry(selection)

    def mouseReleaseEvent(self, event):
        selection = QRect(self._origin, event.pos()).normalized()
        full_image = self.selector.parent().geometry() - QMargins(1, 1, 1, 1)
        self.selector.crop = selection.intersected(full_image).translated(-full_image.topLeft())
        self._rubberband.setGeometry(selection)
        self._rubberband.hide()
        self.repaint()
        self.selector.area_changed.emit()
