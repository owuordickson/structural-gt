import cv2
import json
import numpy as np
from PySide6.QtCore import QObject,Signal, Slot
from PySide6.QtGui import QImage
from PIL import ImageQt  # Import ImageQt for conversion

from gui_tree_model import TreeModel
from gui_table_model import TableModel

from src.StructuralGT.configs.config_loader import load_gtc_configs


class MainController(QObject):
    """Exposes a method to refresh the image in QML"""
    imageChangedSignal = Signal(int, str)
    enableRectangularSelectionSignal = Signal(bool)
    showCroppingToolSignal = Signal(bool)
    performCroppingSignal = Signal(bool)
    adjustBrightnessContrastSignal = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.img_loaded = False
        self.app_active = False
        self.project_loaded = False

        # Initialize flags
        self.error_flag = False
        self.wait_flag = False

        # Create graph objects
        self.analyze_objs = {}
        self.current_obj_id = 0

        # Create Models
        self.graphTreeModel = None
        self.graphPropsTableModel = None
        self.imgListTableModel = None
        self.imgPropsTableModel = None

        # Load Model Data
        self._load_model_data()

        # Load Default Configs
        # self.gui_txt = load_gui_configs()
        # self.configs_data = {}
        ## self.threadpool = QtCore.QThreadPool()
        # self.worker = Worker(0, None)
        # self._load_default_configs()

    def _load_model_data(self):
        """Loads data into models"""
        try:
            with open("assets/data/extract_data.json", "r") as file:
                json_data = json.load(file)
                # self.graphTreeModel.loadData(json_data)  # Assuming TreeModel has a loadData() method
            self.graphTreeModel = TreeModel(json_data)

            data_img_props = data_img_list = data_graph_props = []
            """data_img_props = [
                ["Name", "Invitro.png"],
                ["Width x Height", "500px x 500px"],
                ["Dimensions", "2D"],
                ["Pixel Size", "2nm x 2nm"],
            ]"""
            self.imgPropsTableModel = TableModel(data_img_props)

            """data_img_list = [
                ["Invitro.png"],
                ["Nano-gel.png"],
                ["Bacteria.png"],
            ]"""
            self.imgListTableModel = TableModel(data_img_list)

            """data_graph_props = [
                ["Node Count", "248"],
                ["Edge Count", "306"],
                ["Sub-graph Count", "1"],
                ["Largest-Full Graph Ratio", "100%"],
            ]"""
            self.graphPropsTableModel = TableModel(data_graph_props)
        except Exception as e:
            print(f"Error loading GUI model data: {e}")

    def _load_default_configs(self):
        """
        Initialize all UI configurations.

        :return:
        """

        # 1. Fetch configs
        self.configs_data = load_all_configs()
        options = self.configs_data['main_options']
        options_img = self.configs_data['filter_options']
        options_gte = self.configs_data['extraction_options']
        options_gtc = self.configs_data['sgt_options']

        # 2. Initialize Settings
        self._init_tree(options_gte, options_gtc)

        # 3. Initialize Filter Settings
        self._init_img_filter_settings(options_img)

        # 4. Initialize Binary Settings
        self._init_img_binary_settings(options_img)

        # 5. Initialize Enhance and Compute Tools
        self._init_tools()

        # 6. initialize Image Paths
        self._init_img_path_settings(options)

    @Slot(str, result=bool)
    def load_gte_setting(self, item_name):
        # print(item_name)
        if len(self.analyze_objs) > 0:
            return False
        else:
            # options_gte = self.analyze_objs[self.current_obj_id]
            options_gtc = load_gtc_configs()
            val = options_gtc[item_name]
            print(val)
            # if item_name ==
            return True if val == 1 else False

    @Slot(result=bool)
    def is_image_loaded(self):
        return self.img_loaded

    @Slot(result=bool)
    def is_app_activated(self):
        return self.app_active

    @Slot(result=bool)
    def is_project_loaded(self):
        return self.project_loaded

    @Slot(result=bool)
    def error_occurred(self):
        return self.error_flag

    @Slot(result=bool)
    def in_progress(self):
        return self.wait_flag

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        return "image://imageProvider?t=" + str(np.random.randint(1, 1000))

    @Slot(bool)
    def show_cropping_tool(self, allow_cropping):
        self.showCroppingToolSignal.emit(allow_cropping)

    @Slot(bool)
    def perform_cropping(self, allowed):
        self.performCroppingSignal.emit(allowed)

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
            self.imageChangedSignal.emit(3, "undo")

    @Slot(float, float)
    def brightness_contrast_control(self, brightness, contrast):
        self.adjustBrightnessContrastSignal.emit(brightness, contrast)
        # print(brightness+contrast)

    @Slot(QImage, float, float)
    def adjust_brightness_contrast(self, q_image, brightness, contrast):
        """ Converts QImage to OpenCV format, applies brightness/contrast, and saves. """
        img_pil = ImageQt.fromqimage(q_image)
        img = MainController.q_image_to_cv(img_pil)

        # Apply brightness and contrast adjustments
        brightness = np.clip(brightness, -100, 100)
        contrast = np.clip(contrast, 0.1, 3.0)

        img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)
        processed_path = "assets/processed_image.png"
        cv2.imwrite(processed_path, img)
        print(f"Processed Image Saved: {processed_path}")

        self.imageChangedSignal.emit(3, processed_path)

    @Slot(bool)
    def enable_rectangular_selection(self, enabled):
        self.enableRectangularSelectionSignal.emit(enabled)

    @staticmethod
    def q_image_to_cv(q_image):
        """ Converts QImage to OpenCV format (NumPy array). """
        img = ImageQt.ImageQt(q_image)  # Convert QImage to PIL Image
        img = img.convert("RGB")  # Ensure RGB format
        return np.array(img)[:, :, ::-1]  # Convert PIL Image to OpenCV (BGR)
