import sys
import json
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject

from tree_model import TreeModel
from table_model import TableModel

# Assuming TreeModel and TableModel are properly implemented
class MainWindow(QObject):
    def __init__(self):
        super().__init__()
        self.app = QGuiApplication(sys.argv)
        self.ui_engine = QQmlApplicationEngine()

        # Create Models
        self.graphTreeModel = None
        self.imgPropsTableModel = None

        # Load Data
        self.load()

        # Set Models in QML Context
        self.ui_engine.rootContext().setContextProperty("graphTreeModel", self.graphTreeModel)
        self.ui_engine.rootContext().setContextProperty("imgPropsTableModel", self.imgPropsTableModel)

        # Load UI
        self.ui_engine.load("MainWindow.qml")
        if not self.ui_engine.rootObjects():
            sys.exit(-1)

    def load(self):
        """Loads data into models"""
        try:
            with open("assets/data/extract_data.json", "r") as file:
                json_data = json.load(file)
                # self.graphTreeModel.loadData(json_data)  # Assuming TreeModel has a loadData() method
            self.graphTreeModel = TreeModel(json_data)

            table_data = [
                ["Row 1, Column 1", "Row 1, Column 2"],
                ["Row 2, Column 1", "Row 2, Column 2"],
                ["Row 3, Column 1", "Row 3, Column 2"]
            ]
            self.imgPropsTableModel = TableModel(table_data)

        except Exception as e:
            print(f"Error loading data: {e}")

if __name__ == "__main__":
    main_window = MainWindow()
    sys.exit(main_window.app.exec())