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
from src.StructuralGT.apps.sgtv3.qthread_worker import QThreadWorker, WorkerTasks


class MainController(QObject):
    """Exposes a method to refresh the image in QML"""

    showAlertSignal = Signal(str, str)
    projectOpenedSignal = Signal(str)
    changeImageSignal = Signal(int)
    imageChangedSignal = Signal()
    enableRectangularSelectionSignal = Signal(bool)
    showCroppingToolSignal = Signal(bool)
    showUnCroppingToolSignal = Signal(bool)
    performCroppingSignal = Signal(bool)

    def __init__(self):
        super().__init__()
        self.img_loaded = False
        self.project_open = False

        # Project data
        self.project_data = {"name": "", "file_path": ""}

        # Initialize flags
        self.error_flag = False
        self.wait_flag = False
        self.status_msg = {"title": "", "message": ""}

        # Create graph objects
        self.sgt_objs = {}
        self.current_obj_index = -1
        self.current_img_type = 0

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

        # Create QThreadWorker for long tasks
        self.worker = QThreadWorker(0, None)
        self.worker_tasks = WorkerTasks()

    def load_img_configs(self, sgt_obj):
        """Reload image configuration selections and controls after it is loaded."""
        try:
            # 1.
            options_img = sgt_obj.g_obj.imp.configs
            option_gte = sgt_obj.g_obj.configs
            options_gtc = sgt_obj.configs

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
        try:
            keys_list = list(self.sgt_objs.keys())
            key_at_index = keys_list[self.current_obj_index]
            sgt_obj = self.sgt_objs[key_at_index]
            return sgt_obj
        except IndexError as err:
            logging.exception("No Image Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "No Image Error"
            self.status_msg["message"] = f"No image added! Please import/add an image."
            self.error_flag = True

    def create_sgt_object(self, img_path):
        """
        A function that processes a selected image file and creates an analyzer object with default configurations.

        Args:
            img_path: file path to image

        Returns:
        """

        img_path = self.verify_path(img_path)
        if not img_path:
            return False

        # Try reading the image
        try:
            img_dir, filename = os.path.split(str(img_path))
            out_dir_name = "sgt_files"
            out_dir = os.path.join(img_dir, out_dir_name)
            out_dir = os.path.normpath(out_dir)
            os.makedirs(out_dir, exist_ok=True)

            im_obj = ImageProcessor(str(img_path), out_dir)
            g_obj = GraphAnalyzer(GraphExtractor(im_obj))
            self.sgt_objs[filename] = g_obj
            self.error_flag = False
            return True
        except Exception as err:
            # print(f"Error processing image: {e}")
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "File Error"
            self.status_msg["message"] = f"Error processing image. Try again."
            self.error_flag = True
            return False

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        return "image://imageProvider?t=" + str(np.random.randint(1, 1000))

    @Slot(result=int)
    def get_current_img_type(self):
        return self.current_img_type

    """"@Slot(str, result=bool)
    def get_selected_img_val(self, item_name):
        # print(item_name)
        if len(self.sgt_objs) <= 0:
            return False
        else:
            sgt_obj = self._get_current_obj()
            options_img = sgt_obj.g_obj.imp.configs
            # options_img = load_img_configs()
            val = options_img[item_name]["value"]
            return True if val == 1 else False
    """

    """@Slot(str, result=float)
    def get_selected_img_data(self, item_name):
        # print(item_name)
        if len(self.sgt_objs) <= 0:
            return False
        else:
            sgt_obj = self._get_current_obj()
            options_img = sgt_obj.g_obj.imp.configs
            # options_img = load_img_configs()
            if options_img[item_name]["type"] == "image-filter":
                val = options_img[item_name]["dataValue"]
            else:
                val = options_img[item_name]["value"]
            return val
    """

    @Slot(result=str)
    def get_img_nav_location(self):
        return f"{(self.current_obj_index + 1)} / {len(self.sgt_objs)}"

    @Slot(result=str)
    def get_output_dir(self):
        sgt_obj = self.get_current_obj()
        return f"{sgt_obj.g_obj.imp.output_dir}"

    @Slot(str)
    def set_output_dir(self, folder_path):

        # Convert QML "file:///" path format to a proper OS path
        if folder_path.startswith("file:///"):
            if sys.platform.startswith("win"):  # Windows Fix (remove extra '/')
                folder_path = folder_path[8:]
            else:  # macOS/Linux (remove "file://")
                folder_path = folder_path[7:]
        folder_path = os.path.normpath(folder_path)  # Normalize path

        sgt_obj = self.get_current_obj()
        sgt_obj.g_obj.imp.output_dir = folder_path
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
        self.current_img_type = 0 if (choice == 1 or choice == 5) else choice
        self.changeImageSignal.emit(choice)

    @Slot(int)
    def load_image(self, index):
        try:
            self.current_obj_index = index
            self.imgListTableModel.update_data(self.sgt_objs)
            self.select_img_type(0)
        except Exception as err:
            self.current_obj_index = -1
            # print(f"Error loading GUI model data: {err}")
            self.status_msg["message"] = f"Error loading image. Try again."
            logging.exception("Image Loading Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot(result=bool)
    def load_prev_image(self):
        """Load the previous image in the list into view."""
        if self.current_obj_index > 0:
            pos = self.current_obj_index - 1
            self.load_image(pos)
            # return False if pos == 0 else True
            return True
        else:
            return False

    @Slot(result=bool)
    def load_next_image(self):
        """Load next image in the list into view."""
        if self.current_obj_index < (len(self.sgt_objs) - 1):
            pos = self.current_obj_index + 1
            self.load_image(pos)
            # return False if pos == (len(self.sgt_objs) - 1) else True
            return True
        else:
            return False

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
        # updated_values = [[val["id"], val["value"]] for val in self.gtcListModel.list_data]
        print("GTC Updated Settings:", self.get_current_obj().configs)

    @Slot()
    def apply_gte_changes(self):
        """Retrieve settings from model and send to Python."""
        # opt_gte = {}
        for i in range(self.gteTreeModel.rowCount()):
            parent_index = self.gteTreeModel.index(i, 0)
            print([self.gteTreeModel.data(parent_index, self.gteTreeModel.IdRole),
                   self.gteTreeModel.data(parent_index, self.gteTreeModel.ValueRole)])
            for j in range(self.gteTreeModel.rowCount(parent_index)):
                child_index = self.gteTreeModel.index(j, 0, parent_index)
                print([self.gteTreeModel.data(child_index, self.gteTreeModel.IdRole),
                       self.gteTreeModel.data(child_index, self.gteTreeModel.ValueRole)])
        # print(self.gteTreeModel.data())

    @Slot()
    def apply_img_bin_changes(self):
        """Retrieve settings from model and send to Python."""
        updated_values = [[val["id"], val["value"]] for val in self.imgBinFilterModel.list_data]
        print("Updated Settings:", updated_values)
        # self.select_img_type()
        pass

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
        # self.select_img_type(self.current_img_type)
        pass

    @Slot()
    def apply_microscopy_props_changes(self):
        """Retrieve settings from model and send to Python."""
        updated_values = [[val["id"], val["value"]] for val in self.microscopyPropsModel.list_data]
        print("Updated Settings:", updated_values)

    @Slot(result=bool)
    def display_image(self):
        return self.img_loaded

    @Slot(result=bool)
    def is_project_open(self):
        return self.project_open

    @Slot(result=bool)
    def error_occurred(self):
        return self.error_flag

    @Slot(result=bool)
    def is_running(self):
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
            sgt_obj = self.get_current_obj()
            img = Image.fromarray(sgt_obj.g_obj.imp.img)
            q_img = ImageQt.toqpixmap(img)

            # Convert QImage to PIL Image
            img_pil = ImageQt.fromqimage(q_img)

            # Crop the selected area
            img_pil_crop = img_pil.crop((x, y, x + width, y + height))
            img_cv = ImageProcessor.load_img_from_pil(img_pil_crop)
            sgt_obj.g_obj.imp.img, sgt_obj.g_obj.imp.scale_factor = ImageProcessor.resize_img(512, img_cv)
            sgt_obj.g_obj.reset()

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
            sgt_obj = self.get_current_obj()
            img_cv = sgt_obj.g_obj.imp.img.copy()
            sgt_obj.g_obj.imp.img_mod = ImageProcessor.control_brightness(img_cv, brightness_level, contrast_level)
            # print(f"{brightness_level} brightness and {contrast_level} contrast")
            self.select_img_type(2)
        except Exception as err:
            # print(f"Error adjusting brightness/contrast of image: {err}")
            logging.exception("Image Processing Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot(bool)
    def enable_rectangular_selection(self, enabled):
        self.enableRectangularSelectionSignal.emit(enabled)

    @Slot(result=bool)
    def enable_prev_nav_btn(self):
        if self.current_obj_index == 0:
            return False
        else:
            return True

    @Slot(result=bool)
    def enable_next_nav_btn(self):
        if self.current_obj_index == (len(self.sgt_objs) - 1):
            return False
        else:
            return True

    @Slot(str, result=bool)
    def add_single_image(self, image_path):
        """Verify and validate an image path, use it to create an SGT object and load it in view."""
        is_created = self.create_sgt_object(image_path)
        if is_created:
            pos = (len(self.sgt_objs) - 1)
            self.load_image(pos)
            return True
        return False

    @Slot(str, result=bool)
    def add_multiple_images(self, img_dir_path):
        """
        Verify and validate multiple image paths, use each to create an SGT object, then load the last one in view.
        """

        img_dir_path = self.verify_path(img_dir_path)
        if not img_dir_path:
            return False

        files = os.listdir(img_dir_path)
        files = sorted(files)
        for a_file in files:
            if a_file.endswith(('.tif', '.png', '.jpg', '.jpeg')):
                img_path = os.path.join(str(img_dir_path), a_file)
                self.create_sgt_object(img_path)

        if len(self.sgt_objs) <= 0:
            self.status_msg["title"] = "File Error"
            self.status_msg["message"] = "No workable images found! Files have to be either .tif, .png, or .jpg or .jpeg"
            self.error_flag = True
            return False
        else:
            pos = (len(self.sgt_objs) - 1)
            self.load_image(pos)
            return True

    @Slot(str, str, result=bool)
    def create_sgt_project(self, proj_name, dir_path):
        """Creates a '.sgtproj' inside the selected directory"""

        self.project_open = False
        dir_path = self.verify_path(dir_path)
        if not dir_path:
            return False

        # Create the directory if it doesn't exist
        # results_dir = os.path.join(dir_path, '/results')
        # if not os.path.exists(results_dir):
        #    os.makedirs(results_dir)

        proj_name += '.sgtproj'
        proj_path = os.path.join(str(dir_path), proj_name)

        try:
            # Open the file in write mode ('w'). This will create the file if it doesn't exist
            # and overwrite it if it does.
            with open(proj_path, 'w'):
                pass  # Do nothing, just create the file

            # Update and notify QML
            self.project_data["name"] = proj_name
            self.project_data["path"] = proj_path
            self.project_open = True
            self.projectOpenedSignal.emit(proj_name)
            print(f"File '{proj_name}' created successfully in '{dir_path}'.")
        except Exception as e:
            print(f"An error occurred: {e}")

    @Slot(str, result=bool)
    def open_sgt_project(self, sgt_path):
        """Opens and loads SGT project from the '.sgtproj' file"""

        # Verify path
        sgt_path = self.verify_path(sgt_path)
        if not sgt_path:
            return False

        # Read and load project data and SGT objects
        print(sgt_path)

    def verify_path(self, a_path):
        if not a_path:
            self.status_msg["title"] = "File/Directory Error"
            self.status_msg["message"] = "No folder/file selected."
            self.error_flag = True
            return False

        # Normalize file path
        # Convert QML "file:///" path format to a proper OS path
        if a_path.startswith("file:///"):
            if sys.platform.startswith("win"):  # Windows Fix (remove extra '/')
                a_path = a_path[8:]
            else:  # macOS/Linux (remove "file://")
                a_path = a_path[7:]
        a_path = os.path.normpath(a_path)  # Normalize path

        if not os.path.exists(a_path):
            logging.exception("File/Folder Error: %s", IOError, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "Path Error"
            self.status_msg["message"] = f"File/Folder does not exist - {a_path}. Try again."
            self.error_flag = True
            return False
        return a_path
