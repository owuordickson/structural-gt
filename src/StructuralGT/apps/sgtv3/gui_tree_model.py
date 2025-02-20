from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex


class TreeItem:
    """ Represents a single item in the tree model. """
    def __init__(self, data, parent=None):
        self.parentItem = parent
        self.itemData = data
        self.children = []

    def appendChild(self, child):
        self.children.append(child)

    def child(self, row):
        return self.children[row] if row < len(self.children) else None

    def childCount(self):
        return len(self.children)

    def columnCount(self):
        return 1  # We only display names

    def data(self):
        return self.itemData

    def parent(self):
        return self.parentItem

    def row(self):
        return self.parentItem.children.index(self) if self.parentItem else 0


class TreeModel(QAbstractItemModel):
    """ Provides a hierarchical model for QML TreeView. """
    def __init__(self, data, parent=None):
        super(TreeModel, self).__init__(parent)
        self.rootItem = TreeItem("Root")
        self._setupModelData(data, self.rootItem)

    def _setupModelData(self, data, parent):
        """ Recursively adds items to the model from JSON data. """
        for key, value in data.items():
            item = TreeItem(key, parent)
            parent.appendChild(item)
            if isinstance(value, dict):
                self._setupModelData(value, item)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0

        parent_item = self.rootItem if not parent.isValid() else parent.internalPointer()
        return parent_item.childCount()

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return ""
        if role == Qt.DisplayRole:
            return str(index.internalPointer().data())  # Ensure valid string for display
        return ""

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        parent_item = self.rootItem if not parent.isValid() else parent.internalPointer()
        child_item = parent_item.child(row)

        if child_item:
            return self.createIndex(row, column, child_item)

        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self.rootItem or parentItem is None:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)