import os
import sys
import cv2
import logging
import numpy as np
from PySide6.QtCore import QObject,Signal, Slot
from PIL import Image, ImageQt  # Import ImageQt for conversion

from gui_tree_model import TreeModel
from gui_table_model import TableModel
from gui_list_model import CheckBoxModel

from src.StructuralGT.configs.config_loader import load_gtc_configs, load_gte_configs, load_img_configs
from src.StructuralGT.SGT.image_processor import ImageProcessor
from src.StructuralGT.SGT.graph_extractor import GraphExtractor
from src.StructuralGT.SGT.graph_analyzer import GraphAnalyzer


class MainController(QObject):
    """Exposes a method to refresh the image in QML"""
    changeImageSignal = Signal(int)
    imageChangedSignal = Signal()
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
        self.status_msg = {"title": "", "message": ""}

        # Create graph objects
        self.analyze_objs = {}
        self.current_obj_index = 0

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

    def _get_current_obj(self):
        keys_list = list(self.analyze_objs.keys())
        key_at_index = keys_list[self.current_obj_index]
        a_obj = self.analyze_objs[key_at_index]
        return a_obj

    def _load_model_data(self):
        """Loads data into models"""
        try:
            # 1.
            options_img = load_img_configs()
            option_gte = load_gte_configs()
            options_gtc = load_gtc_configs()

            # 2.
            graph_options = [v for v in option_gte.values() if v["type"] == "graph-extraction"]
            # file_options = [v for v in option_gte.values() if v["type"] == "file-options"]

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
            a_obj = self._get_current_obj()
            options_img = a_obj.g_obj.imp.configs
            # options_img = load_img_configs()
            val = options_img[item_name]["value"]
            return True if val == 1 else False

    @Slot(str, result=float)
    def load_img_setting_val(self, item_name):
        # print(item_name)
        if len(self.analyze_objs) <= 0:
            return False
        else:
            a_obj = self._get_current_obj()
            options_img = a_obj.g_obj.imp.configs
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
            a_obj = self._get_current_obj()
            options_gtc = a_obj.configs
            # options_gtc = load_gtc_configs()
            val = options_gtc[item_name]["value"]
            # print(val)
            return True if val == 1 else False

    @Slot(result=bool)
    def display_image(self):
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

    @Slot( int, int, int, int)
    def crop_image(self, x, y, width, height):
        """Crop image using PIL and save it."""
        try:
            a_obj = self._get_current_obj()
            img = Image.fromarray(a_obj.g_obj.imp.img)
            q_img = ImageQt.toqpixmap(img)

            # Convert QImage to PIL Image
            img_pil = ImageQt.fromqimage(q_img)

            # Crop the selected area
            img_pil_crop = img_pil.crop((x, y, x + width, y + height))
            img_cv = ImageProcessor.load_img_from_pil(img_pil_crop)
            a_obj.g_obj.imp.img, a_obj.g_obj.imp.scale_factor = ImageProcessor.resize_img(512, img_cv)
            a_obj.g_obj.reset()

            # Emit signal to update UI with new image
            self.changeImageSignal.emit(1)
            self.showCroppingToolSignal.emit(False)
        except Exception as err:
            # print(f"Error cropping image: {err}")
            logging.exception("Cropping Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot(bool)
    def undo_cropping(self, undo: bool = True):
        if undo:
            self.changeImageSignal.emit(4)

    @Slot( float, float)
    def adjust_brightness_contrast(self, brightness_level, contrast_level):
        """ Converts QImage to OpenCV format, applies brightness/contrast, and saves. """
        try:
            a_obj = self._get_current_obj()
            img_cv = a_obj.g_obj.imp.img.copy()
            a_obj.g_obj.imp.img_mod = ImageProcessor.control_brightness(img_cv, brightness_level, contrast_level)

            """img_cv = a_obj.g_obj.imp.img
            # Apply brightness and contrast adjustments
            brightness = np.clip(brightness_level, -100, 100)
            contrast = np.clip(contrast_level, 0.1, 3.0)
            img_cv = cv2.convertScaleAbs(img_cv, alpha=contrast, beta=brightness)
            a_obj.g_obj.imp.img_mod = img_cv"""

            self.changeImageSignal.emit(2)
        except Exception as err:
            # print(f"Error adjusting brightness/contrast of image: {err}")
            logging.exception("Image Processing Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot(bool)
    def enable_rectangular_selection(self, enabled):
        self.enableRectangularSelectionSignal.emit(enabled)

    @Slot(str)
    def process_selected_file(self, img_path):
        """"""
        if not img_path:
            self.status_msg["title"] = "File Error"
            self.status_msg["message"] = "No file selected."
            self.error_flag = True
            return

        # Convert QML "file:///" path format to a proper OS path
        if img_path.startswith("file:///"):
            if sys.platform.startswith("win"):  # Windows Fix (remove extra '/')
                img_path = img_path[8:]
            else:  # macOS/Linux (remove "file://")
                img_path = img_path[7:]
        img_path = os.path.normpath(img_path)  # Normalize path

        # Check if file exists
        if not os.path.exists(img_path):
            self.status_msg["title"] = "File Error"
            self.status_msg["message"] = f"File does not exist - {img_path}. Try again."
            self.error_flag = True
            return

        # Try reading the image
        try:
            img_dir, filename = os.path.split(img_path)
            out_dir_name = "sgt_files"
            out_dir = os.path.join(img_dir, out_dir_name)
            out_dir = os.path.normpath(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            self.analyze_objs = {}
            im_obj = ImageProcessor(img_path, out_dir)
            g_obj = GraphAnalyzer(GraphExtractor(im_obj))
            self.analyze_objs[filename] = g_obj
            self.current_obj_index = 0
            self.error_flag = False

            self.changeImageSignal.emit(0)

        except Exception as err:
            # print(f"Error processing image: {e}")
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "File Error"
            self.status_msg["message"] = f"Error loading/processing image. Try again."
            self.error_flag = True
