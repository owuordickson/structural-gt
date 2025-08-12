from PyQt6.QtCore import Qt, QAbstractListModel, QModelIndex, QPersistentModelIndex

# Define a simple QAbstractListModel
class CheckBoxModel(QAbstractListModel):
    IdRole = Qt.ItemDataRole.UserRole + 1
    TextRole = Qt.ItemDataRole.UserRole + 3
    ValueRole = Qt.ItemDataRole.UserRole + 4

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.list_data = data

    def rowCount(self, parent=None):
        return len(self.list_data)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.list_data):
            return None
        item = self.list_data[index.row()]
        if role == self.IdRole:
            return item["id"]
        elif role == self.TextRole:
            return item["text"]
        elif role == self.ValueRole:
            return item["value"]
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.DisplayRole):
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
        return False

    def reset_data(self, new_data):
        self.beginResetModel()
        self.list_data = new_data
        self.endResetModel()
        self.dataChanged.emit(self.index(0,0), self.index(len(new_data), 0),
                              [self.IdRole, self.TextRole, self.ValueRole])

    def roleNames(self):
        return {
            self.IdRole: b"id",
            self.TextRole: b"text",
            self.ValueRole: b"value"
        }
