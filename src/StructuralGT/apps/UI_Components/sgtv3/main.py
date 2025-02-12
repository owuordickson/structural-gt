# This Python file uses the following encoding: utf-8
import json
import sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from tree_model import TreeModel

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # Load JSON data
    with open("assets/data/extract_data.json", "r") as file:
        extract_data = json.load(file)
    # with open("assets/data/gt_data.json", "r") as file:
    #    gt_data = json.load(file)

    extr_tree_model = TreeModel(extract_data)
    engine.rootContext().setContextProperty("extractModel", extr_tree_model)
    # gt_tree_model = TreeModel(gt_data)
    # engine.rootContext().setContextProperty("gtModel", gt_tree_model)

    engine.load("MainWindow.qml")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
