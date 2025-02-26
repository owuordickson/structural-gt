import os
import sys
import logging
import numpy as np
from PIL import Image, ImageQt  # ImageQt for conversion
from PySide6.QtCore import QObject,Signal,Slot
from matplotlib.backends.backend_pdf import PdfPages

from gui_tree_model import TreeModel
from gui_table_model import TableModel
from gui_list_model import CheckBoxModel

from src.StructuralGT import __version__
from src.StructuralGT.SGT.image_processor import ImageProcessor
from src.StructuralGT.SGT.graph_extractor import GraphExtractor
from src.StructuralGT.SGT.graph_analyzer import GraphAnalyzer
from src.StructuralGT.apps.sgtv3.qthread_worker import QThreadWorker, WorkerTask


class MainController(QObject):
    """Exposes a method to refresh the image in QML"""

    showAlertSignal = Signal(str, str)
    errorSignal = Signal(str)
    updateProgressSignal = Signal(int, str)
    taskTerminatedSignal = Signal(bool, list)
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
        self.current_obj_index = 0
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
        self.worker_task = WorkerTask()

    def load_configs_to_models(self, sgt_obj):
        """Load image configuration selections and controls after it is loaded."""
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

            self.microscopyPropsModel.reset_data(img_properties)
            self.imgPropsTableModel.reset_data(sgt_obj.g_obj.imp.props)
            self.graphPropsTableModel.reset_data(sgt_obj.g_obj.props)
        except Exception as err:
            # print(f"Error loading GUI model data: {err}")
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "Fatal Error"
            self.status_msg["message"] = f"Error loading image configurations! Close app and try again."
            self.error_flag = True

    def update_configs_models(self, sgt_obj):
        """Reload image configuration selections and controls after it is loaded."""
        try:
            option_gte = sgt_obj.g_obj.configs
            options_gtc = sgt_obj.configs

            graph_options = [v for v in option_gte.values() if v["type"] == "graph-extraction"]
            # file_options = [v for v in option_gte.values() if v["type"] == "file-options"]

            self.gtcListModel.reset_data(list(options_gtc.values()))
            self.gteTreeModel.reset_data(graph_options)

            self.imgPropsTableModel.reset_data(sgt_obj.g_obj.imp.props)
            self.graphPropsTableModel.reset_data(sgt_obj.g_obj.props)
        except Exception as err:
            # print(f"Error loading GUI model data: {err}")
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "Fatal Error"
            self.status_msg["message"] = f"Error re-loading image configurations! Close app and try again."
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
            self.load_configs_to_models(g_obj)
            self.error_flag = False
            return True
        except Exception as err:
            # print(f"Error processing image: {e}")
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "File Error"
            self.status_msg["message"] = f"Error processing image. Try again."
            self.error_flag = True
            return False

    def _handle_progress_update(self, value: int, msg: str):
        """"""
        progress_val = 0 if value < 0 else value
        print(str(progress_val) + "%: " + msg)
        logging.info(str(progress_val) + "%: " + msg, extra={'user': 'SGT Logs'})
        if value >= 0:
            self.updateProgressSignal.emit(progress_val, msg)
        else:
            self.errorSignal.emit(msg)

    def _handle_finished(self, success_val: bool, result: None|list|GraphExtractor|GraphAnalyzer):
        """"""
        self.error_flag = success_val
        self.wait_flag = False
        if not success_val:
            if type(result) is list:
                logging.info(result[0] + ": " + result[1], extra={'user': 'SGT Logs'})
                self.status_msg["title"] = result[0]
                self.status_msg["message"] = result[1]
                self.taskTerminatedSignal.emit(success_val, result)
            elif type(result) is GraphAnalyzer:
                self.write_to_pdf(result)
        else:
            if type(result) is GraphExtractor:
                self._handle_progress_update(100, "Graph extracted successfully!")
                sgt_obj = self.get_current_obj()
                sgt_obj.g_obj = result
                # Load image superimposed with graph
                self.select_img_type(4)
                # Send task termination signal to QML
                self.taskTerminatedSignal.emit(success_val, [])
            elif type(result) is GraphAnalyzer:
                self._handle_progress_update(100, "GT PDF successfully generated!")
                self.taskTerminatedSignal.emit(1, ["GT calculations completed", "The image's GT parameters have been "
                                                                                "calculated. Check out generated PDF in "
                                                                                "'Output Dir'."])
            elif type(result) is dict:
                self._handle_progress_update(100, "All GT PDF successfully generated!")
                self.taskTerminatedSignal.emit(1, ["All GT calculations completed", "GT parameters of all "
                                                                                    "images have been calculated. Check "
                                                                                    "out all the generated PDFs in "
                                                                                    "'Output Dir'."])
            else:
                self.taskTerminatedSignal.emit(success_val, [])

    @Slot(result=str)
    def get_sgt_version(self):
        """"""
        # Copyright (C) 2024, the Regents of the University of Michigan.
        return f"StructuralGT v{__version__}"

    @Slot(result=list)
    def get_alert_message(self):
        """"""
        if self.error_flag:
            return [self.status_msg["title"], self.status_msg["message"]]
        else:
            return []

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        return "image://imageProvider?t=" + str(np.random.randint(1, 1000))

    @Slot(result=int)
    def get_current_img_type(self):
        return self.current_img_type

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
    def select_img_type(self, choice=None):
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
        choice = self.current_img_type if choice is None else choice
        self.current_img_type = 0 if (choice == 1 or choice == 5) else choice
        self.changeImageSignal.emit(choice)

    @Slot(int)
    def load_image(self, index=None):
        try:
            self.current_obj_index = index if index is not None else self.current_obj_index
            self.imgListTableModel.update_data(self.sgt_objs)
            self.select_img_type()
        except Exception as err:
            self.current_obj_index = 0
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
    def apply_img_bin_changes(self):
        """Retrieve settings from model and send to Python."""
        # updated_values = [[val["id"], val["value"]] for val in self.imgBinFilterModel.list_data]
        # print("Updated Settings:", updated_values)
        self.select_img_type()

    @Slot()
    def apply_img_filter_changes(self):
        """Retrieve settings from model and send to Python."""
        """updated_values = []
        for item in self.imgFilterModel.list_data:
            try:
                val = [item["id"], item["value"], item["dataValue"]]
            except KeyError:
                val = [item["id"], item["value"]]
            updated_values.append(val)
        print("Updated Settings:", updated_values)"""
        self.select_img_type()

    @Slot()
    def run_extract_graph(self):
        """Retrieve settings from model and send to Python."""
        """for i in range(self.gteTreeModel.rowCount()):
            parent_index = self.gteTreeModel.index(i, 0)
            print([self.gteTreeModel.data(parent_index, self.gteTreeModel.IdRole),
                   self.gteTreeModel.data(parent_index, self.gteTreeModel.ValueRole)])
            for j in range(self.gteTreeModel.rowCount(parent_index)):
                child_index = self.gteTreeModel.index(j, 0, parent_index)
                print([self.gteTreeModel.data(child_index, self.gteTreeModel.IdRole),
                       self.gteTreeModel.data(child_index, self.gteTreeModel.ValueRole)])"""
        self.worker_task = WorkerTask()
        try:
            self.wait_flag = True
            sgt_obj = self.get_current_obj()

            self.worker = QThreadWorker(func=self.worker_task.task_extract_graph, args=(sgt_obj.g_obj,))
            self.worker_task.inProgressSignal.connect(self._handle_progress_update)
            self.worker_task.taskFinishedSignal.connect(self._handle_finished)
            self.worker.start()
        except Exception as err:
            print(f"An error occurred: {err}")
            logging.exception("Graph Extraction Error: %s", IOError, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Fatal error occurred! Close the app and try again.")
            self.worker_task.taskFinishedSignal.emit(-1, ["Graph Extraction Error",
                                                          "Fatal error while trying to extract graph. "
                                                          "Close the app and try again."])

    @Slot()
    def run_graph_analyzer(self):
        """Retrieve settings from model and send to Python."""
        # updated_values = [[val["id"], val["value"]] for val in self.gtcListModel.list_data]
        # print("GTC Updated Settings:", self.get_current_obj().configs)
        self.worker_task = WorkerTask()
        try:
            self.wait_flag = True
            sgt_obj = self.get_current_obj()

            self.worker = QThreadWorker(func=self.worker_task.task_compute_gt, args=(sgt_obj,))
            self.worker_task.inProgressSignal.connect(self._handle_progress_update)
            self.worker_task.taskFinishedSignal.connect(self._handle_finished)
            self.worker.start()
        except Exception as err:
            print(f"An error occurred: {err}")
            logging.exception("GT Computation Error: %s", IOError, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Fatal error occurred! Close the app and try again.")
            self.worker_task.taskFinishedSignal.emit(-1, ["GT Computation Error",
                                                          "Fatal error while trying calculate GT parameters. "
                                                          "Close the app and try again."])

    @Slot()
    def run_multi_graph_analyzer(self):
        """"""
        self.worker_task = WorkerTask()
        try:
            self.wait_flag = True

            self.worker = QThreadWorker(func=self.worker_task.task_compute_multi_gt, args=(self.sgt_objs,))
            self.worker_task.inProgressSignal.connect(self._handle_progress_update)
            self.worker_task.taskFinishedSignal.connect(self._handle_finished)
            self.worker.start()
        except Exception as err:
            print(f"An error occurred: {err}")
            logging.exception("GT Computation Error: %s", IOError, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Fatal error occurred! Close the app and try again.")
            self.worker_task.taskFinishedSignal.emit(-1, ["GT Computation Error",
                                                          "Fatal error while trying calculate GT parameters. "
                                                          "Close the app and try again."])

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
    def is_task_running(self):
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
            # pos = (len(self.sgt_objs) - 1)
            self.load_image()
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
            # pos = (len(self.sgt_objs) - 1)
            self.load_image()
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
        except Exception as err:
            print(f"An error occurred: {err}")
            logging.exception("Create Project Error: %s", IOError, extra={'user': 'SGT Logs'})
            self.status_msg["title"] = "Create Project Error"
            self.status_msg["message"] = f"Fatal error while trying to create SGT project. Close the app and try again."
            self.error_flag = True

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

    def write_to_pdf(self, sgt_obj):
        """
        Write results to PDF file.
        Args:
            sgt_obj:

        Returns:

        """
        try:
            self._handle_progress_update(98, "Writing PDF...")

            filename, output_location = sgt_obj.g_obj.imp.create_filenames()
            pdf_filename = filename + "_SGT_results.pdf"
            pdf_file = os.path.join(output_location, pdf_filename)
            with (PdfPages(pdf_file) as pdf):
                for fig in sgt_obj.plot_figures:
                    pdf.savefig(fig)
            sgt_obj.g_obj.save_files()

            self._handle_finished(True, sgt_obj)
        except Exception as err:
            print(err)
            logging.exception("GT Computation Error: %s", IOError, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Error occurred while trying to write to PDF.")
            # self.worker_task.taskFinishedSignal.emit(-1, ["GT Computation Error", "Error occurred while trying to write "
            #                                                                      "to PDF. Run GT computations again."])
