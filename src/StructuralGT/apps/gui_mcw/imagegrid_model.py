from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QPersistentModelIndex


class ImageGridModel(QAbstractListModel):
    SelectedRole = Qt.ItemDataRole.UserRole + 20
    ImageRole = Qt.ItemDataRole.UserRole + 21

    def __init__(self, img_data, parent=None):
        super().__init__(parent)
        self._images = img_data

    def rowCount(self, parent=None):
        return len(self._images)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._images):
            return None

        if role == self.ImageRole:
            return self._images[index.row()]
        return None

    def roleNames(self):
        return {
            self.SelectedRole: b"selected",
            self.ImageRole: b"image",
        }
