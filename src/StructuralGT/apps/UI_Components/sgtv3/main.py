import sys
import json
import numpy as np
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject,Signal, Slot
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtQuick import QQuickImageProvider

from PIL import ImageQt  # Import ImageQt for conversion

from tree_model import TreeModel
from table_model import TableModel


class ImageProvider(QQuickImageProvider):

    def __init__(self, img_controller):
        super().__init__(QQuickImageProvider.Pixmap)
        # self.images = {}  # Store images with their IDs
        self._image_path = ""
        self.pixmap = QPixmap()
        self.pixmap_original = QPixmap()
        self.pixmap_cropped = QPixmap()
        self.img_controller = img_controller
        self.img_controller.imageChangedSignal.connect(self.handle_change_image)

    """def requestImage(self, id, requested_size, size):
        if id in self.images:
            pixmap = self.images[id]
            size.setWidth(pixmap.width())
            size.setHeight(pixmap.height())
            return pixmap
        return QPixmap()

    def add_image(self, id, pixmap):
        self.images[id] = pixmap"""

    def set_image(self, image_path: str, option: str =""):
        self._image_path = image_path
        if option == "crop":
            self.pixmap_cropped.load(image_path)
        else:
            self.pixmap_original.load(image_path)
        self.img_controller.img_loaded = False
        # print(image_path)

    def select_image(self, option: str=""):
        if option == "crop":
            self.pixmap = self.pixmap_cropped
        else:
            self.pixmap = self.pixmap_original
        self.img_controller.img_loaded = True

    def requestPixmap(self, img_id, requested_size, size):
        # print(img_id)
        return self.pixmap

    def handle_change_image(self, src, img_path):
        if src == 1:  # '0'-Original, '1'-Crop, '2'-Undo crop,  ignore '3' - will make function recursive
            self.set_image(img_path, "crop")
            self.select_image("crop")
            self.img_controller.imageChangedSignal.emit(2, img_path)
        elif src == 2:
            self.select_image("")



class ImageController(QObject):
    """Exposes a method to refresh the image in QML"""
    imageChangedSignal = Signal(int, str)
    enableRectangularSelectionSignal = Signal(bool)
    showCroppingToolSignal = Signal(bool)
    performCroppingSignal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.img_loaded = False

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        return "image://imageProvider?t=" + str(np.random.randint(1, 1000))

    @Slot(QImage, int, int, int, int)
    def crop_image(self, q_image, x, y, width, height):
        """Crop image using PIL and save it."""
        try:
            # Convert QPixmap to QImage
            # q_image = pixmap.toImage()

            # Convert QImage to PIL Image
            img_pil = ImageQt.fromqimage(q_image)

            # Crop the selected area
            img_cropped = img_pil.crop((x, y, x + width, y + height))

            # Save cropped image
            cropped_path = "assets/cropped_image.png"
            img_cropped.save(cropped_path)
            # print(f"Cropped image saved: {cropped_path}")

            # Emit signal to update UI with new image
            self.imageChangedSignal.emit(1, cropped_path)
            self.showCroppingToolSignal.emit(False)
        except Exception as e:
            print(f"Error cropping image: {e}")

    @Slot(bool)
    def undo_cropping(self, undo: bool = True):
        if undo:
            self.imageChangedSignal.emit(2, "undo")

    @Slot(result=bool)
    def is_image_loaded(self):
        return self.img_loaded

    @Slot(bool)
    def enable_rectangular_selection(self, enabled):
        self.enableRectangularSelectionSignal.emit(enabled)

    @Slot(bool)
    def perform_cropping(self, allowed):
        self.performCroppingSignal.emit(allowed)

    @Slot(bool)
    def show_cropping_tool(self, allow_cropping):
        self.showCroppingToolSignal.emit(allow_cropping)


# Assuming TreeModel and TableModel are properly implemented
class MainWindow(QObject):
    def __init__(self):
        super().__init__()
        self.app = QGuiApplication(sys.argv)
        self.ui_engine = QQmlApplicationEngine()

        # Create Models
        self.graphTreeModel = None
        self.imgPropsTableModel = None
        self.graphPropsTableModel = None

        # Register Controller for Dynamic Updates
        self.image_controller = ImageController()
        # Register Image Provider
        self.image_provider = ImageProvider(self.image_controller)

        # Load Data
        self.load()

        # Set Models in QML Context
        self.ui_engine.rootContext().setContextProperty("graphTreeModel", self.graphTreeModel)
        self.ui_engine.rootContext().setContextProperty("imgPropsTableModel", self.imgPropsTableModel)
        self.ui_engine.rootContext().setContextProperty("graphPropsTableModel", self.graphPropsTableModel)
        self.ui_engine.rootContext().setContextProperty("imageController", self.image_controller)
        self.ui_engine.addImageProvider("imageProvider", self.image_provider)

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

            data_img_props = [
                ["Name", "Invitro.png"],
                ["Width x Height", "500px x 500px"],
                ["Dimensions", "2D"],
                ["Pixel Size", "2nm x 2nm"],
            ]
            self.imgPropsTableModel = TableModel(data_img_props)

            data_graph_props = [
                ["Node Count", "248"],
                ["Edge Count", "306"],
                ["Sub-graph Count", "1"],
                ["Largest-Full Graph Ratio", "100%"],
            ]
            self.graphPropsTableModel = TableModel(data_graph_props)

            # Images
            img_path = "assets/icons/graph_icon.png"
            self.image_provider.set_image(img_path)
            self.image_provider.select_image("")
            # pixmap = QPixmap(img_path)
            # self.image_provider.add_image("test_image", pixmap)
        except Exception as e:
            print(f"Error loading data: {e}")

if __name__ == "__main__":
    main_window = MainWindow()
    sys.exit(main_window.app.exec())
