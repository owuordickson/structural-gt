from PySide6.QtCore import Qt, QAbstractListModel, QModelIndex, QPersistentModelIndex

# Define a simple QAbstractListModel
class CheckBoxModel(QAbstractListModel):
    IdRole = Qt.ItemDataRole.UserRole + 1
    TypeRole = Qt.ItemDataRole.UserRole + 2
    TextRole = Qt.ItemDataRole.UserRole + 3
    ValueRole = Qt.ItemDataRole.UserRole + 4
    DataIdRole = Qt.ItemDataRole.UserRole + 5
    DataValueRole = Qt.ItemDataRole.UserRole + 6
    MinValueRole = Qt.ItemDataRole.UserRole + 7
    MaxValueRole = Qt.ItemDataRole.UserRole + 8
    StepSizeRole = Qt.ItemDataRole.UserRole + 9
    VisibleRole = Qt.ItemDataRole.UserRole + 10
    ToolTipRole = Qt.ItemDataRole.ToolTipRole + 11

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.list_data = data

    def rowCount(self, parent=None):
        return len(self.list_data)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.list_data):
            return None
        item = self.list_data[index.row()]

        # Map roles → dictionary keys
        role_map = {
            self.IdRole: "id",
            self.TypeRole: "type",
            self.TextRole: "text",
            self.ValueRole: "value",
            self.DataIdRole: "dataId",
            self.DataValueRole: "dataValue",
            self.MinValueRole: "minValue",
            self.MaxValueRole: "maxValue",
            self.StepSizeRole: "stepSize",
            self.VisibleRole: "visible",
            self.ToolTipRole: "tooltip",
        }

        key = role_map.get(role, None)
        if key is None:
            return None

        # Safe return: return value if key exists, else None
        return item.get(key, None)

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
        if role == self.DataValueRole:
            self.list_data[index.row()]["dataValue"] = value
            self.dataChanged.emit(index, index, [role])
            return True
        return False

    def reset_data(self, new_data):
        self.beginResetModel()
        self.list_data = new_data
        self.endResetModel()
        self.dataChanged.emit(self.index(0,0), self.index(len(new_data), 0),
                              [self.IdRole, self.TypeRole, self.TextRole, self.ValueRole, self.DataIdRole,
                               self.DataValueRole, self.MinValueRole, self.MaxValueRole, self.StepSizeRole,
                               self.VisibleRole, self.ToolTipRole])

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
            self.VisibleRole: b"visible",
            self.ToolTipRole: b"tooltip"
        }