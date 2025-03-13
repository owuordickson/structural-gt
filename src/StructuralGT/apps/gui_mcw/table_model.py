import base64
from PIL import Image, ImageQt  # Import ImageQt for conversion
from PIL.ImageQt import QIODevice
from PySide6.QtCore import QByteArray, QBuffer
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt

# Define a simple table model
class TableModel(QAbstractTableModel):
    SelectedRole = Qt.UserRole + 10

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.itemData = data
        self.imageCache = {}
        self.selected_index = -1  # Track selected row

    def rowCount(self, parent=QModelIndex()):
        return len(self.itemData) if self.itemData else 0

    def columnCount(self, parent=QModelIndex()):
        return len(self.itemData[0]) if self.itemData else 0

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount()):
            return None
        if role == Qt.DisplayRole:
            return self.itemData[index.row()][index.column()]
        elif role == Qt.DecorationRole:
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
        new_keys = list(analyze_objs.keys())
        if not new_keys:
            return  # No data to add

        start_row = len(self.itemData)
        self.beginResetModel()
        self.itemData = []
        self.imageCache = {}

        for key in new_keys:
            self.itemData.append([key])  # Store the key
            a_obj = analyze_objs[key]
            img_cv = a_obj.g_obj.imp.img_2d  # Assuming OpenCV image format
            img_pil = Image.fromarray(img_cv)  # Convert to PIL Image
            pixmap = ImageQt.toqpixmap(img_pil)  # Convert to QPixmap

            # Convert QPixmap to QImage
            q_image = pixmap.toImage()

            # Convert QImage to base64 string
            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            buffer.open(QIODevice.WriteOnly)
            q_image.save(buffer, "PNG")  # Save QImage to buffer as PNG
            base64_data = base64.b64encode(byte_array.data()).decode("utf-8")  # Encode to base64

            self.imageCache[key] = base64_data  # Store base64 string
            # self.imageCache[key] = pixmap  # Store QPixmap in cache
        self.endResetModel()
        # Emit dataChanged signal to notify QML
        self.dataChanged.emit(self.index(start_row, 0), self.index(len(self.itemData) - 1, 0))

    def roleNames(self):
        return {
            Qt.DisplayRole: b"text",
            Qt.DecorationRole: b"thumbnail",
            self.SelectedRole: b"selected",
        }

    def headerData(self, section, orientation, role=Qt.DisplayRole):
         if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return f"Column {section + 1}"
         return super().headerData(section, orientation, role)