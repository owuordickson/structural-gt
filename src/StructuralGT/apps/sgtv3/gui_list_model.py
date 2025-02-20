from PySide6.QtCore import Qt, QAbstractListModel

# Define a simple QAbstractListModel
class CheckBoxModel(QAbstractListModel):
    TextRole = Qt.UserRole + 1
    IdRole = Qt.UserRole + 2
    DataIdRole = Qt.UserRole + 3
    DataValueRole = Qt.UserRole + 4
    MinValueRole = Qt.UserRole + 5
    MaxValueRole = Qt.UserRole + 6
    StepSizeRole = Qt.UserRole + 7

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.list_data = data

    def rowCount(self, parent=None):
        return len(self.list_data)

    def data(self, index, role):
        if not index.isValid() or index.row() >= len(self.list_data):
            return None
        item = self.list_data[index.row()]
        if role == self.TextRole:
            return item["text"]
        elif role == self.IdRole:
            return item["id"]
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

    def roleNames(self):
        return {
            self.TextRole: b"text",
            self.IdRole: b"id",
            self.DataIdRole: b"dataId",
            self.DataValueRole: b"dataValue",
            self.MinValueRole: b"minValue",
            self.MaxValueRole: b"maxValue",
            self.StepSizeRole: b"stepSize",
        }