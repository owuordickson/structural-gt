from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt

class TreeItem:
    def __init__(self, data, parent=None):
        self.data = data
        self.parent_item = parent
        self.child_items = []

    def append_child(self, item):
        self.child_items.append(item)

    def child(self, row):
        return self.child_items[row]

    def child_count(self):
        return len(self.child_items)

    def column_count(self):
        return len(self.data)

    def data(self, column):
        try:
            return self.data[column]
        except IndexError:
            return None

    def parent(self):
        return self.parent_item

    def row(self):
        if self.parent_item:
            return self.parent_item.child_items.index(self)
        return 0

class TreeModel(QAbstractItemModel):
    def __init__(self, root, parent=None):
        super().__init__(parent)
        self.root_item = TreeItem(["Root"])
        self.setup_model_data(root, self.root_item)

    def columnCount(self, parent=QModelIndex()):
        return self.root_item.column_count()

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        item = self.get_item(index)
        if role == Qt.DisplayRole:
            return item.data(index.column())
        return None

    def flags(self, index):
        if not index.isValid():
            return Qt.NoItemFlags
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable

    def get_item(self, index):
         if index.isValid():
            item = index.internalPointer()
            if item:
                return item
         return self.root_item

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.root_item.data(section)
        return None

    def index(self, row, column, parent=QModelIndex()):
        if not self.hasIndex(row, column, parent):
            return QModelIndex()

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        child_item = parent_item.child(row)
        if child_item:
            return self.createIndex(row, column, child_item)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        child_item = self.get_item(index)
        parent_item = child_item.parent()

        if parent_item == self.root_item:
            return QModelIndex()

        return self.createIndex(parent_item.row(), 0, parent_item)

    def rowCount(self, parent=QModelIndex()):
        if parent.column() > 0:
            return 0

        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()

        return parent_item.child_count()

    def setup_model_data(self, data, parent):
        for name, children in data.items():
            item_data = [name]
            item = TreeItem(item_data, parent)
            parent.append_child(item)
            if children:
                self.setup_model_data(children, item)
