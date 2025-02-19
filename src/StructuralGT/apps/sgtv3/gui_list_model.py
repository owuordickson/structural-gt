from PySide6.QtCore import Qt, QAbstractListModel

# Define a simple QAbstractListModel
class CheckBoxModel(QAbstractListModel):
    TextRole = Qt.UserRole + 1
    IdRole = Qt.UserRole + 2

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
        return None

    def roleNames(self):
        return {
            self.TextRole: b"text",
            self.IdRole: b"id",
        }