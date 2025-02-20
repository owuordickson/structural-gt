import cv2
import json
import numpy as np
from PySide6.QtCore import QObject,Signal, Slot
from PySide6.QtGui import QImage
from PIL import ImageQt  # Import ImageQt for conversion

from gui_tree_model import TreeModel
from gui_table_model import TableModel
from gui_list_model import CheckBoxModel

from src.StructuralGT.configs.config_loader import load_gtc_configs, load_gte_configs, load_img_configs


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
        self.graphPropsTableModel = None
        self.imgListTableModel = None
        self.imgPropsTableModel = None

        self.gteTreeModel = None
        self.gtcListModel = None
        self.imgBinFilterModel = None
        self.imgFilterModel = None
        self.imgControlModel = None
        self.microscopyPropsModel = None

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
            # 1.
            options_img = load_img_configs()
            option_gte = load_gte_configs()
            options_gtc = load_gtc_configs()

            # 2.
            graph_options = [v for v in option_gte.values() if v["type"] == "graph-extraction"]
            file_options = [v for v in option_gte.values() if v["type"] == "file-options"]

            bin_filters = [v for v in options_img.values() if v["type"] == "binary-filter"]
            img_filters = [v for v in options_img.values() if v["type"] == "image-filter"]
            img_controls = [v for v in options_img.values() if v["type"] == "image-control"]
            img_properties = [v for v in options_img.values() if v["type"] == "image-property"]

            self.gtcListModel = CheckBoxModel(list(options_gtc.values()))
            self.gteTreeModel = TreeModel(graph_options)
            self.imgBinFilterModel = CheckBoxModel(bin_filters)
            self.imgFilterModel = CheckBoxModel(img_filters)
            self.imgControlModel = CheckBoxModel(img_controls)
            self.microscopyPropsModel = CheckBoxModel(img_properties)

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

    @Slot(str, result=bool)
    def load_img_setting(self, item_name):
        # print(item_name)
        if len(self.analyze_objs) <= 0:
            return False
        else:
            options_img = self.analyze_objs[self.current_obj_id].g_obj.imp.configs
            # options_img = load_img_configs()
            val = options_img[item_name]["value"]
            return True if val == 1 else False

    @Slot(str, result=float)
    def load_img_setting_val(self, item_name):
        # print(item_name)
        if len(self.analyze_objs) <= 0:
            return False
        else:
            options_img = self.analyze_objs[self.current_obj_id].g_obj.imp.configs
            # options_img = load_img_configs()
            if options_img[item_name]["type"] == "image-filter":
                val = options_img[item_name]["dataValue"]
            else:
                val = options_img[item_name]["value"]
            return val

    @Slot(result=bool)
    def load_gte_setting(self):
        if len(self.analyze_objs) > 0:
            return False
        else:
            # options_gte = self.analyze_objs[self.current_obj_id].g_obj.configs
            options_gte = load_gte_configs()

    @Slot(str, result=bool)
    def load_gtc_setting(self, item_name):
        # print(item_name)
        if len(self.analyze_objs) <= 0:
            return False
        else:
            options_gtc = self.analyze_objs[self.current_obj_id].configs
            # options_gtc = load_gtc_configs()
            val = options_gtc[item_name]["value"]
            # print(val)
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
