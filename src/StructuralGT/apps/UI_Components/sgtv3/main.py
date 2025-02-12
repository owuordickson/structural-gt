# This Python file uses the following encoding: utf-8
import json
import sys
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from tree_model import TreeModel

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    # code to read from json
    extract_data = json.load(open('assets/data/extract_data.json', 'r'))
    gt_data = json.load(open('assets/data/gt_data.json', 'r'))

    extract_tree_model = TreeModel(extract_data)
    context = engine.rootContext()
    context.setContextProperty("extract_data", extract_data)

    gt_tree_model = TreeModel(gt_data)
    context = engine.rootContext()
    context.setContextProperty("gt_data", gt_data)

    engine.load("MainWindow.qml")
    if not engine.rootObjects():
        sys.exit(-1)
    sys.exit(app.exec())
