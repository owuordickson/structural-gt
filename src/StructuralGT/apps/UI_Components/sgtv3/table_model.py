import sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

# Define a simple table model
class TableModel(QAbstractTableModel):
    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.itemData = data

    def rowCount(self, parent=QModelIndex()):
        return len(self.itemData)

    def columnCount(self, parent=QModelIndex()):
        return len(self.itemData[0]) if self.itemData else 0

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None
        if role == Qt.DisplayRole:
            return self.itemData[index.row()][index.column()]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
         if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return f"Column {section + 1}"
         return super().headerData(section, orientation, role)