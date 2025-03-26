import numpy as np
from PySide6.QtCore import Qt, QAbstractListModel

from ...SGT.sgt_utils import get_cv_base64, get_plt_base64


class ImageGridModel(QAbstractListModel):
    IdRole = Qt.ItemDataRole.UserRole + 1
    SelectedRole = Qt.ItemDataRole.UserRole + 20
    ImageRole = Qt.ItemDataRole.UserRole + 21

    def __init__(self, img_lst, parent=None):
        super().__init__(parent)
        if len(img_lst) == 0:
            self._image_data = []
            return

        if type(img_lst[0]) is np.ndarray:
            self._image_data = [{"id": i, "image": get_cv_base64(img_lst[i]), "selected": 0} for i in range(len(img_lst))]
        else:
            self._image_data = [{"id": i, "image": get_plt_base64(img_lst[i]), "selected": 0} for i in range(len(img_lst))]

    def rowCount(self, parent=None):
        return len(self._image_data)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._image_data):
            return None

        if role == self.IdRole:
            return self._image_data[index.row()]["id"]
        elif role == self.ImageRole:
            return self._image_data[index.row()]["image"]
        elif role == self.SelectedRole:
            return self._image_data[index.row()]["selected"]
        return None

    def setData(self, index, value, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._image_data):
            return False

        if role == self.SelectedRole:
            self._image_data[index.row()]["selected"] = value
            self.dataChanged.emit(index, index, [role])
            return True
        if role == self.ImageRole:
            self._image_data[index.row()]["image"] = value
            self.dataChanged.emit(index, index, [role])
            return True

    def reset_data(self, new_data):
        """ Resets the data to be displayed. """
        self._image_data = new_data
        if new_data is None:
            return
        self.beginResetModel()

        if type(new_data[0]) is np.ndarray:
            self._image_data = [{"id": i, "image": get_cv_base64(new_data[i]), "selected": 0} for i in range(len(new_data))]
        else:
            self._image_data = [{"id": i, "image": get_plt_base64(new_data[i]), "selected": 0} for i in range(len(new_data))]

        self.endResetModel()
        self.dataChanged.emit(self.index(0, 0), self.index(len(new_data), 0),
                              [self.IdRole, self.ImageRole, self.SelectedRole])

    def roleNames(self):
        return {
            self.IdRole: b"id",
            self.ImageRole: b"image",
            self.SelectedRole: b"selected",
        }
