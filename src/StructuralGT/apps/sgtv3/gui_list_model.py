from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QPersistentModelIndex

# Define a simple QAbstractListModel
class CheckBoxModel(QAbstractListModel):
    IdRole = Qt.UserRole + 1
    TypeRole = Qt.UserRole + 2
    TextRole = Qt.UserRole + 3
    ValueRole = Qt.UserRole + 4
    DataIdRole = Qt.UserRole + 5
    DataValueRole = Qt.UserRole + 6
    MinValueRole = Qt.UserRole + 7
    MaxValueRole = Qt.UserRole + 8
    StepSizeRole = Qt.UserRole + 9

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.list_data = data

    def rowCount(self, parent=None):
        return len(self.list_data)

    def data(self, index, role):
        if not index.isValid() or index.row() >= len(self.list_data):
            return None
        item = self.list_data[index.row()]
        if role == self.IdRole:
            return item["id"]
        elif role == self.TypeRole:
            return item["type"]
        elif role == self.TextRole:
            return item["text"]
        elif role == self.ValueRole:
            return item["value"]
        elif role == self.DataIdRole:
            return item["dataId"]
        elif role == self.DataValueRole:
            return item["dataValue"]
        elif role == self.MinValueRole:
            return item["minValue"]
        elif role == self.MaxValueRole:
            return item["maxValue"]
        elif role == self.StepSizeRole:
            return item["stepSize"]
        return None

    def setData(self, index, value, role):
        """

        Args:
            index (QModelIndex | QPersistentModelIndex):
            value (int|float):
            role (int):
        """
        if not index.isValid() or index.row() >= len(self.list_data):
            return False

        if role == self.ValueRole:
            self.list_data[index.row()]["value"] = value
            self.dataChanged.emit(index, index, [role])
            return True
        if role == self.DataValueRole:
            self.list_data[index.row()]["dataValue"] = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def roleNames(self):
        return {
            self.IdRole: b"id",
            self.TypeRole: b"type",
            self.TextRole: b"text",
            self.ValueRole: b"value",
            self.DataIdRole: b"dataId",
            self.DataValueRole: b"dataValue",
            self.MinValueRole: b"minValue",
            self.MaxValueRole: b"maxValue",
            self.StepSizeRole: b"stepSize",
        }