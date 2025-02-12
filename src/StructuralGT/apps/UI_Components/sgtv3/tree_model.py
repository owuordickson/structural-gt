import sys
import json
from PySide6.QtWidgets import QApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import Qt, QAbstractItemModel, QModelIndex# , QVariant


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
        parentItem = self.rootItem if not parent.isValid() else parent.internalPointer()
        return parentItem.childCount()

    def columnCount(self, parent=QModelIndex()):
        return 1

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return ""

        return str(index.internalPointer().data())  # Ensure it returns a string

    def index(self, row, column, parent=QModelIndex()):
        if parent.isValid():
            parentItem = parent.internalPointer()
        else:
            parentItem = self.rootItem

        childItem = parentItem.child(row)
        if childItem:
            return self.createIndex(row, column, childItem)
        return QModelIndex()

    def parent(self, index):
        if not index.isValid():
            return QModelIndex()

        childItem = index.internalPointer()
        parentItem = childItem.parent()

        if parentItem == self.rootItem or parentItem is None:
            return QModelIndex()

        return self.createIndex(parentItem.row(), 0, parentItem)