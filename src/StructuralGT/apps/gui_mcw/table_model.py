from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

from ...SGT.sgt_utils import get_pixmap

# Define a simple table model
class TableModel(QAbstractTableModel):
    SelectedRole = Qt.ItemDataRole.UserRole + 20

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.itemData = data
        self.imageCache = {}
        self.selected_index = -1  # Track selected row

    def rowCount(self, parent=QModelIndex()):
        return len(self.itemData) if self.itemData else 0

    def columnCount(self, parent=QModelIndex()):
        return len(self.itemData[0]) if self.itemData else 0

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None
        if role == Qt.ItemDataRole.DisplayRole:
            return self.itemData[index.row()][index.column()]
        elif role == Qt.ItemDataRole.DecorationRole:
            if len(self.imageCache) <= 0:
                return None
            image_name = self.itemData[index.row()][index.column()]
            return self.imageCache[image_name]
        elif role == self.SelectedRole:
            return index.row() == self.selected_index  # True if selected
        return None

    def set_selected(self, row):
        if 0 <= row < len(self.itemData):
            old_index = self.selected_index
            self.selected_index = row
            if old_index != -1:
                self.dataChanged.emit(self.index(old_index, 0), self.index(old_index, 0), [self.SelectedRole])
            self.dataChanged.emit(self.index(row, 0), self.index(row, 0), [self.SelectedRole])

    def reset_data(self, item_data):
        self.beginResetModel()
        self.imageCache = {}
        self.itemData = item_data
        self.endResetModel()
        self.dataChanged.emit(self.index(0, 0), self.index(len(self.itemData) - 1, 0))

    def update_data(self, analyze_objs):
        """Updates model with new images from analyze_objs"""
        self.beginResetModel()
        new_keys = list(analyze_objs.keys())
        if not new_keys:
            # No data to add/update
            self.itemData = []
            self.imageCache = {}
            self.endResetModel()
            # Emit dataChanged signal to notify QML
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.itemData) - 1, 0))
            return

        start_row = len(self.itemData)
        # self.beginResetModel()
        self.itemData = []
        self.imageCache = {}

        for key in new_keys:
            self.itemData.append([key])  # Store the key
            a_obj = analyze_objs[key]
            img_cv = a_obj.g_obj.imp.img_2d  # Assuming OpenCV image format
            base64_data = get_pixmap(img_cv)

            self.imageCache[key] = base64_data  # Store base64 string
            # self.imageCache[key] = pixmap  # Store QPixmap in cache
        self.endResetModel()
        # Emit dataChanged signal to notify QML
        self.dataChanged.emit(self.index(start_row, 0), self.index(len(self.itemData) - 1, 0))

    def roleNames(self):
        return {
            Qt.ItemDataRole.DisplayRole: b"text",
            Qt.ItemDataRole.DecorationRole: b"thumbnail",
            self.SelectedRole: b"selected",
        }

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
         if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return f"Column {section + 1}"
         return super().headerData(section, orientation, role)