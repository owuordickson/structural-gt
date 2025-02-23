import os
import sys
import logging
import numpy as np
from PIL import Image, ImageQt  # Import ImageQt for conversion
from PySide6.QtCore import QObject,Signal, Slot

from gui_tree_model import TreeModel
from gui_table_model import TableModel
from gui_list_model import CheckBoxModel

from src.StructuralGT.SGT.image_processor import ImageProcessor
from src.StructuralGT.SGT.graph_extractor import GraphExtractor
from src.StructuralGT.SGT.graph_analyzer import GraphAnalyzer


class MainController(QObject):
    """Exposes a method to refresh the image in QML"""
    changeImageSignal = Signal(int)
    imageChangedSignal = Signal()
    enableRectangularSelectionSignal = Signal(bool)
    showCroppingToolSignal = Signal(bool)
    showUnCroppingToolSignal = Signal(bool)
    performCroppingSignal = Signal(bool)

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
        self.current_obj_index = -1

        # Create Models
        self.imgListTableModel = TableModel([])
        self.imgPropsTableModel = TableModel([])
        self.graphPropsTableModel = TableModel([])
        self.microscopyPropsModel = CheckBoxModel([])

        self.gteTreeModel = TreeModel([])
        self.gtcListModel = CheckBoxModel([])
        self.imgBinFilterModel = CheckBoxModel([])
        self.imgFilterModel = CheckBoxModel([])
        self.imgControlModel = CheckBoxModel([])

    def load_img_configs(self, a_obj):
        """Reload image configuration selections and controls after it is loaded."""
        try:
            # 1.
            options_img = a_obj.g_obj.imp.configs
            option_gte = a_obj.g_obj.configs
            options_gtc = a_obj.configs

            # 2.
            graph_options = [v for v in option_gte.values() if v["type"] == "graph-extraction"]
            # file_options = [v for v in option_gte.values() if v["type"] == "file-options"]

            bin_filters = [v for v in options_img.values() if v["type"] == "binary-filter"]
            img_filters = [v for v in options_img.values() if v["type"] == "image-filter"]
            img_controls = [v for v in options_img.values() if v["type"] == "image-control"]
            img_properties = [v for v in options_img.values() if v["type"] == "image-property"]

            self.gtcListModel.reset_data(list(options_gtc.values()))
            self.gteTreeModel.reset_data(graph_options)
            self.imgBinFilterModel.reset_data(bin_filters)
            self.imgFilterModel.reset_data(img_filters)
            self.imgControlModel.reset_data(img_controls)

            data_img_props = data_graph_props = []
            """data_img_props = [
                ["Name", "Invitro.png"],
                ["Width x Height", "500px x 500px"],
                ["Dimensions", "2D"],
                ["Pixel Size", "2nm x 2nm"],
            ]"""
            """data_graph_props = [
                ["Node Count", "248"],
                ["Edge Count", "306"],
                ["Sub-graph Count", "1"],
                ["Largest-Full Graph Ratio", "100%"],
            ]"""

            self.microscopyPropsModel.reset_data(img_properties)
            self.imgPropsTableModel.reset_data(data_img_props)
            self.graphPropsTableModel .reset_data(data_graph_props)

        except Exception as err:
            # print(f"Error loading GUI model data: {err}")
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "Fatal Error"
            self.status_msg["message"] = f"Error loading image configurations! Close app and try again."
            self.error_flag = True

    def get_current_obj(self):
        keys_list = list(self.analyze_objs.keys())
        key_at_index = keys_list[self.current_obj_index]
        a_obj = self.analyze_objs[key_at_index]
        return a_obj

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        return "image://imageProvider?t=" + str(np.random.randint(1, 1000))

    """"@Slot(str, result=bool)
    def get_selected_img_val(self, item_name):
        # print(item_name)
        if len(self.analyze_objs) <= 0:
            return False
        else:
            a_obj = self._get_current_obj()
            options_img = a_obj.g_obj.imp.configs
            # options_img = load_img_configs()
            val = options_img[item_name]["value"]
            return True if val == 1 else False
    """

    """@Slot(str, result=float)
    def get_selected_img_data(self, item_name):
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
    """

    @Slot(result=str)
    def get_img_nav_location(self):
        return f"{(self.current_obj_index + 1)} / {len(self.analyze_objs)}"

    @Slot(result=str)
    def get_output_dir(self):
        a_obj = self.get_current_obj()
        return f"{a_obj.g_obj.imp.output_dir}"

    @Slot(str)
    def set_output_dir(self, folder_path):

        # Convert QML "file:///" path format to a proper OS path
        if folder_path.startswith("file:///"):
            if sys.platform.startswith("win"):  # Windows Fix (remove extra '/')
                folder_path = folder_path[8:]
            else:  # macOS/Linux (remove "file://")
                folder_path = folder_path[7:]
        folder_path = os.path.normpath(folder_path)  # Normalize path

        a_obj = self.get_current_obj()
        a_obj.g_obj.imp.output_dir = folder_path
        self.imageChangedSignal.emit()

    @Slot(int)
    def select_img_type(self, choice):
        """
            '0' - Original image
            '1' - Cropped image
            '2' - Processed image
            '3' - Binary image
            '4' - Extracted graph
            '5' - Undo crop
        Args:
            choice:
        Returns:
        """
        self.changeImageSignal.emit(choice)

    @Slot(int)
    def load_image(self, index):
        try:
            self.current_obj_index = index
            self.imgListTableModel.update_data(self.analyze_objs)
            self.select_img_type(0)
        except Exception as err:
            self.current_obj_index = -1
            # print(f"Error loading GUI model data: {err}")
            self.status_msg["message"] = f"Error loading image. Try again."
            logging.exception("Image Loading Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot()
    def apply_img_ctrl_changes(self):
        """Retrieve settings from model and send to Python."""
        updated_values = [[val["id"], val["value"]] for val in self.imgControlModel.list_data]
        brightness = 0
        contrast = 0
        for item in updated_values:
            if item[0] == "brightness_level":
                brightness = item[1]
            if item[0] == "contrast_level":
                contrast = item[1]
        # print("Updated Settings:", updated_values)
        self.adjust_brightness_contrast(brightness, contrast)

    @Slot()
    def apply_gtc_changes(self):
        """Retrieve settings from model and send to Python."""
        updated_values = [[val["id"], val["value"]] for val in self.gtcListModel.list_data]
        print("Updated Settings:", updated_values)

    @Slot()
    def apply_img_bin_changes(self):
        """Retrieve settings from model and send to Python."""
        updated_values = [[val["id"], val["value"]] for val in self.imgBinFilterModel.list_data]
        print("Updated Settings:", updated_values)

    @Slot()
    def apply_img_filter_changes(self):
        """Retrieve settings from model and send to Python."""
        updated_values = []
        for item in self.imgFilterModel.list_data:
            try:
                val = [item["id"], item["value"], item["dataValue"]]
            except KeyError:
                val = [item["id"], item["value"]]
            updated_values.append(val)
        print("Updated Settings:", updated_values)

    @Slot()
    def apply_microscopy_props_changes(self):
        """Retrieve settings from model and send to Python."""
        updated_values = [[val["id"], val["value"]] for val in self.microscopyPropsModel.list_data]
        print("Updated Settings:", updated_values)

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
            a_obj = self.get_current_obj()
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
            self.select_img_type(1)
            self.showCroppingToolSignal.emit(False)
            self.showUnCroppingToolSignal.emit(True)
        except Exception as err:
            # print(f"Error cropping image: {err}")
            logging.exception("Cropping Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot(bool)
    def undo_cropping(self, undo: bool = True):
        if undo:
            self.select_img_type(5)
            self.showUnCroppingToolSignal.emit(False)

    @Slot( float, float)
    def adjust_brightness_contrast(self, brightness_level, contrast_level):
        """ Converts QImage to OpenCV format, applies brightness/contrast, and saves. """
        try:
            a_obj = self.get_current_obj()
            img_cv = a_obj.g_obj.imp.img.copy()
            a_obj.g_obj.imp.img_mod = ImageProcessor.control_brightness(img_cv, brightness_level, contrast_level)
            # print(f"{brightness_level} brightness and {contrast_level} contrast")
            self.select_img_type(2)
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

            # self.analyze_objs = {}
            im_obj = ImageProcessor(img_path, out_dir)
            g_obj = GraphAnalyzer(GraphExtractor(im_obj))
            self.analyze_objs[filename] = g_obj
            index = (len(self.analyze_objs) - 1)
            self.error_flag = False
            self.load_image(index)
        except Exception as err:
            # print(f"Error processing image: {e}")
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "File Error"
            self.status_msg["message"] = f"Error processing image. Try again."
            self.error_flag = True
