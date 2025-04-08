import os
import sys
import logging
import pickle
import numpy as np
from typing import TYPE_CHECKING, Optional
from PySide6.QtCore import QObject,Signal,Slot
from matplotlib.backends.backend_pdf import PdfPages

from .imagegrid_model import ImageGridModel

if TYPE_CHECKING:
    # False at run time, only for type checker
    from _typeshed import SupportsWrite

from .tree_model import TreeModel
from .table_model import TableModel
from .checkbox_model import CheckBoxModel
from .qthread_worker import QThreadWorker, WorkerTask

from ... import __version__
from ...SGT.network_processor import NetworkProcessor, ALLOWED_IMG_EXTENSIONS
from ...SGT.graph_extractor import GraphExtractor, COMPUTING_DEVICE
from ...SGT.graph_analyzer import GraphAnalyzer
from ...SGT.sgt_utils import get_cv_base64


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
        self.allow_auto_scale = True

        # Project data
        self.project_data = {"name": "", "file_path": ""}

        # Initialize flags
        self.wait_flag = False

        # Create graph objects
        self.sgt_objs = {}
        self.selected_sgt_obj_index = 0
        self.selected_img_type = 0

        # Create Models
        self.imgThumbnailModel = TableModel([])
        self.imagePropsModel = TableModel([])
        self.graphPropsModel = TableModel([])
        self.microscopyPropsModel = CheckBoxModel([])

        self.gteTreeModel = TreeModel([])
        self.gtcListModel = CheckBoxModel([])
        self.exportGraphModel = CheckBoxModel([])
        self.imgBatchModel = CheckBoxModel([])
        self.imgControlModel = CheckBoxModel([])
        self.imgBinFilterModel = CheckBoxModel([])
        self.imgFilterModel = CheckBoxModel([])
        self.imgScaleOptionModel = CheckBoxModel([])
        self.saveImgModel = CheckBoxModel([])
        self.img3dGridModel = ImageGridModel([])

        # Create QThreadWorker for long tasks
        self.worker = QThreadWorker(0, None)
        self.worker_task = WorkerTask()

    def update_img_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload image configuration selections and controls from saved dict to QML gui_mcw after the image is loaded.

            :param sgt_obj: a GraphAnalyzer object with all saved user-selected configurations.
        """
        try:
            ntwk_p = sgt_obj.ntwk_p
            first_index = next(iter(ntwk_p.selected_images), None)  # 1st selected image
            first_index = first_index if first_index is not None else 0  # first image if None
            options_img = ntwk_p.images[first_index].configs

            img_controls = [v for v in options_img.values() if v["type"] == "image-control"]
            bin_filters = [v for v in options_img.values() if v["type"] == "binary-filter"]
            img_filters = [v for v in options_img.values() if v["type"] == "image-filter"]
            img_properties = [v for v in options_img.values() if v["type"] == "image-property"]
            file_options = [v for v in options_img.values() if v["type"] == "file-options"]

            self.imgControlModel.reset_data(img_controls)
            self.imgBinFilterModel.reset_data(bin_filters)
            self.imgFilterModel.reset_data(img_filters)
            self.microscopyPropsModel.reset_data(img_properties)
            self.saveImgModel.reset_data(file_options)
        except Exception as err:
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Fatal Error", "Error re-loading image configurations! Close app and try again.")

    def update_graph_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload graph configuration selections and controls from saved dict to QML gui_mcw.
        Args:
            sgt_obj: a GraphAnalyzer object with all saved user-selected configurations.

        Returns:

        """
        try:
            ntwk_p = sgt_obj.ntwk_p
            graph_obj = ntwk_p.graph_obj
            option_gte = graph_obj.configs
            options_gtc = sgt_obj.configs

            graph_options = [v for v in option_gte.values() if v["type"] == "graph-extraction"]
            file_options = [v for v in option_gte.values() if v["type"] == "file-options"]
            options_scaling = ntwk_p.scaling_options
            batch_list = [{"id": f"batch_{i}", "text": f"Image Batch {i+1}", "value": i}
                          for i in range(len(sgt_obj.ntwk_p.image_batches))]

            self.imgBatchModel.reset_data(batch_list)
            self.imgScaleOptionModel.reset_data(options_scaling)

            self.gteTreeModel.reset_data(graph_options)
            self.exportGraphModel.reset_data(file_options)
            self.gtcListModel.reset_data(list(options_gtc.values()))

            self.imagePropsModel.reset_data(sgt_obj.ntwk_p.props)
            self.graphPropsModel.reset_data(graph_obj.props)
        except Exception as err:
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Fatal Error", "Error re-loading image configurations! Close app and try again.")

    def get_selected_sgt_obj(self):
        try:
            keys_list = list(self.sgt_objs.keys())
            key_at_index = keys_list[self.selected_sgt_obj_index]
            sgt_obj = self.sgt_objs[key_at_index]
            return sgt_obj
        except IndexError:
            logging.info("No Image Error: Please import/add an image.", extra={'user': 'SGT Logs'})
            # self.showAlertSignal.emit("No Image Error", "No image added! Please import/add an image.")
            return None

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

            ntwk_p = NetworkProcessor(str(img_path), out_dir, self.allow_auto_scale)
            sgt_obj = GraphAnalyzer(ntwk_p)
            self.sgt_objs[filename] = sgt_obj
            self.update_img_models(sgt_obj)
            self.update_graph_models(sgt_obj)
            return True
        except Exception as err:
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File Error", "Error processing image. Try again.")
            return False

    def delete_sgt_object(self, index=None):
        """
        Delete SGT Obj stored at specified index (if not specified, get the current index).
        """
        del_index = index if index is not None else self.selected_sgt_obj_index
        if 0 <= del_index < len(self.sgt_objs):  # Check if index exists
            keys_list = list(self.sgt_objs.keys())
            key_at_del_index = keys_list[self.selected_sgt_obj_index]
            # Delete the object at index
            del self.sgt_objs[key_at_del_index]
            # Update Data
            img_list, img_cache = self.get_thumbnail_list()
            self.imgThumbnailModel.update_data(img_list, img_cache)
            self.selected_sgt_obj_index = 0
            self.load_image()
            self.imageChangedSignal.emit()

    def save_project_data(self):
        """
        A handler function that handles saving project data.
        Returns: True on success, False otherwise.

        """
        if not self.project_open:
            return False
        try:
            file_path = self.project_data["file_path"]
            with open(file_path, 'wb') as project_file:  # type: Optional[SupportsWrite[bytes]]
                pickle.dump(self.sgt_objs, project_file)
            return True
        except Exception as err:
            logging.exception("Project Saving Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Save Error", "Unable to save project data. Close app and try again.")
            return False

    def get_thumbnail_list(self):
        """
        Get names and base64 data of images to be used in Project List thumbnails.
        """
        keys_list = list(self.sgt_objs.keys())
        if len(keys_list) <= 0:
            return None, None
        item_data = []
        image_cache = {}
        for key in keys_list:
            item_data.append([key])  # Store the key
            sgt_obj = self.sgt_objs[key]
            ntwk_p = sgt_obj.ntwk_p
            img_cv = ntwk_p.images[0].img_2d  # First image, assuming OpenCV image format
            base64_data = get_cv_base64(img_cv)
            image_cache[key] = base64_data  # Store base64 string
        return item_data, image_cache

    def get_selected_images(self):
        """
        Get selected images.
        """
        sgt_obj = self.get_selected_sgt_obj()
        ntwk_p = sgt_obj.ntwk_p
        sel_images = [ntwk_p.images[i] for i in ntwk_p.selected_images]
        return sel_images

    def verify_path(self, a_path):
        if not a_path:
            logging.info("No folder/file selected.", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File/Directory Error", "No folder/file selected.")
            return False

        # print(a_path)
        # Convert QML "file:///" path format to a proper OS path
        if a_path.startswith("file:///"):
            if sys.platform.startswith("win"):  # Windows Fix (remove extra '/')
                a_path = a_path[8:]
            else:  # macOS/Linux (remove "file://")
                a_path = a_path[7:]
        # Normalize path
        a_path = os.path.normpath(a_path)
        # print(a_path)

        # Convert to a proper file system path
        """import urllib.parse
        a_path = urllib.parse.urlparse(a_path).path

        # If running on Windows, remove leading '/'
        if os.name == "nt":
            a_path = a_path.lstrip("/")  # Remove leading slash for Windows
        print(a_path)"""

        if not os.path.exists(a_path):
            logging.exception("Path Error: %s", IOError, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Path Error", f"File/Folder in {a_path} does not exist. Try again.")
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

            filename, output_location = sgt_obj.ntwk_p.get_filenames()
            pdf_filename = filename + "_SGT_results.pdf"
            pdf_file = os.path.join(output_location, pdf_filename)
            with (PdfPages(pdf_file) as pdf):
                for fig in sgt_obj.plot_figures:
                    pdf.savefig(fig)

            self._handle_finished(True, sgt_obj)
        except Exception as err:
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Error occurred while trying to write to PDF.")

    def _handle_progress_update(self, value: int, msg: str):
        """
        Handler function for progress updates for ongoing tasks.
        Args:
            value:
            msg:

        Returns:

        """
        progress_val = 0 if value < 0 else value
        logging.info(str(progress_val) + "%: " + msg, extra={'user': 'SGT Logs'})
        if value >= 0:
            self.updateProgressSignal.emit(progress_val, msg)
        else:
            self.errorSignal.emit(msg)

    def _handle_finished(self, success_val: bool, result: None|list|GraphExtractor|GraphAnalyzer):
        """
        Handler function for sending updates/signals on termination of tasks.
        Args:
            success_val:
            result:

        Returns:

        """
        self.wait_flag = False
        if not success_val:
            if type(result) is list:
                logging.info(result[0] + ": " + result[1], extra={'user': 'SGT Logs'})
                self.taskTerminatedSignal.emit(success_val, result)
            elif type(result) is GraphAnalyzer:
                self.write_to_pdf(result)
        else:
            if type(result) is NetworkProcessor:
                self._handle_progress_update(100, "Graph extracted successfully!")
                sgt_obj = self.get_selected_sgt_obj()
                sgt_obj.ntwk_p = result
                # Load image superimposed with graph
                self.select_img_type(4)
                # Send task termination signal to QML
                self.taskTerminatedSignal.emit(success_val, [])
            elif type(result) is GraphAnalyzer:
                self._handle_progress_update(100, "GT PDF successfully generated!")
                self.taskTerminatedSignal.emit(True, ["GT calculations completed", "The image's GT parameters have been "
                                                                                "calculated. Check out generated PDF in "
                                                                                "'Output Dir'."])
            elif type(result) is dict:
                self._handle_progress_update(100, "All GT PDF successfully generated!")
                self.taskTerminatedSignal.emit(True, ["All GT calculations completed", "GT parameters of all "
                                                                                    "images have been calculated. Check "
                                                                                    "out all the generated PDFs in "
                                                                                    "'Output Dir'."])
            else:
                self.taskTerminatedSignal.emit(success_val, [])

    @Slot(result=str)
    def get_sgt_version(self):
        """"""
        # Copyright (C) 2024, the Regents of the University of Michigan.
        return f"StructuralGT v{__version__}, Computing: {COMPUTING_DEVICE}"

    @Slot(result=str)
    def get_about_details(self):
        about_app = (
            f"A software tool that allows graph theory analysis of nano-structures. This is a modified version "
            "of StructuralGT initially proposed by Drew A. Vecchio, DOI: "
            "<html><a href='https://pubs.acs.org/doi/10.1021/acsnano.1c04711'>10.1021/acsnano.1c04711</a></html><html><br/><br/></html>"
            "Contributors:<html><br/></html>"
            "1. Nicolas Kotov<html><br/></html>"
            "2. Dickson Owuor<html><br/></html>"
            "3. Alain Kadar<html><br/><br/></html>"
            "Documentation:<html><br/></html>"
            "<html><a href='https://structural-gt.readthedocs.io'>structural-gt.readthedocs.io</a></html><html><br/><br/></html>"
            f"{self.get_sgt_version()}<html><br/></html>"
            "Copyright (C) 2018-2025, The Regents of the University of Michigan.<html><br/></html>"
            "License: GPL GNU v3<html><br/></html>")
        return about_app

    @Slot(str, result=str)
    def get_file_extensions(self, option):
        if option == "img":
            pattern_string = ' '.join(ALLOWED_IMG_EXTENSIONS)
            return f"Image files ({pattern_string})"
        elif option == "proj":
            return "Project files (*.sgtproj)"
        else:
            return ""

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        unique_num = self.selected_sgt_obj_index + self.selected_img_type + np.random.randint(low=21, high=1000)
        return "image://imageProvider/" + str(unique_num)

    @Slot(result=bool)
    def is_img_3d(self):
        sgt_obj = self.get_selected_sgt_obj()
        if sgt_obj is None:
            return False
        is_3d = True if len(sgt_obj.ntwk_p.images) > 1 else False
        return is_3d

    @Slot(result=int)
    def get_selected_img_batch(self):
        sgt_obj = self.get_selected_sgt_obj()
        return sgt_obj.ntwk_p.selected_image_batch

    @Slot(result=int)
    def get_selected_img_type(self):
        return self.selected_img_type

    @Slot(result=str)
    def get_img_nav_location(self):
        return f"{(self.selected_sgt_obj_index + 1)} / {len(self.sgt_objs)}"

    @Slot(result=str)
    def get_output_dir(self):
        sgt_obj = self.get_selected_sgt_obj()
        if sgt_obj is None:
            return ""
        return f"{sgt_obj.ntwk_p.output_dir}"

    @Slot(result=bool)
    def get_auto_scale(self):
        return self.allow_auto_scale

    @Slot(int)
    def set_selected_thumbnail(self, row_index):
        """Change color of list item to gray if it is the active image"""
        self.imgThumbnailModel.set_selected(row_index)

    @Slot(int)
    def delete_selected_thumbnail(self, img_index):
        """Delete the selected image from list."""
        self.delete_sgt_object(img_index)

    @Slot(str)
    def set_output_dir(self, folder_path):

        # Convert QML "file:///" path format to a proper OS path
        if folder_path.startswith("file:///"):
            if sys.platform.startswith("win"):  # Windows Fix (remove extra '/')
                folder_path = folder_path[8:]
            else:  # macOS/Linux (remove "file://")
                folder_path = folder_path[7:]
        folder_path = os.path.normpath(folder_path)  # Normalize path

        # Update for all sgt_objs
        key_list = list(self.sgt_objs.keys())
        for key in key_list:
            sgt_obj = self.sgt_objs[key]
            sgt_obj.ntwk_p.output_dir = folder_path
        self.imageChangedSignal.emit()

    @Slot(bool)
    def set_auto_scale(self, auto_scale):
        """Set the auto-scale parameter for each image."""
        self.allow_auto_scale = auto_scale

    @Slot(int)
    def select_img_batch(self, batch_index=-1):
        if batch_index < 0:
            return

        try:
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.select_image_batch(batch_index)
            self.select_img_type()
        except Exception as err:
            logging.exception("Batch Change Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Image Batch Error", f"Error encountered while trying to access batch "
                                                           f"{batch_index}. Restart app and try again.")

    @Slot(int)
    def select_img_type(self, choice=None):
        """
            '0' - Original image
            '2' - Processed image
            '3' - Binary image
            '4' - Extracted graph
        Args:
            choice:
        Returns:
        """
        choice = self.selected_img_type if choice is None else choice
        self.selected_img_type = 0 if (choice == 1 or choice == 5) else choice
        self.changeImageSignal.emit(choice)

    @Slot(int)
    def load_image(self, index=None):
        try:
            self.selected_sgt_obj_index = index if index is not None else self.selected_sgt_obj_index
            img_list, img_cache = self.get_thumbnail_list()
            self.imgThumbnailModel.update_data(img_list, img_cache)
            self.imgThumbnailModel.set_selected(self.selected_sgt_obj_index)
            self.select_img_type()
        except Exception as err:
            self.delete_sgt_object()
            self.selected_sgt_obj_index = 0
            logging.exception("Image Loading Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Image Error", "Error loading image. Try again.")

    @Slot(result=bool)
    def load_prev_image(self):
        """Load the previous image in the list into view."""
        if self.selected_sgt_obj_index > 0:
            # pos = self.current_obj_index - 1
            # self.load_image(pos)
            self.selected_sgt_obj_index = self.selected_sgt_obj_index - 1
            self.update_img_models(self.get_selected_sgt_obj())
            self.load_image(self.selected_sgt_obj_index)
            # return False if pos == 0 else True
            return True
        return False

    @Slot(result=bool)
    def load_next_image(self):
        """Load next image in the list into view."""
        if self.selected_sgt_obj_index < (len(self.sgt_objs) - 1):
            self.selected_sgt_obj_index = self.selected_sgt_obj_index + 1
            self.update_img_models(self.get_selected_sgt_obj())
            self.load_image(self.selected_sgt_obj_index)
            # return False if pos == (len(self.sgt_objs) - 1) else True
            return True
        else:
            return False

    @Slot()
    def apply_img_ctrl_changes(self):
        """Retrieve settings from model and send to Python."""
        try:
            sel_images = self.get_selected_images()
            if len(sel_images) <= 0:
                return
            for val in self.imgControlModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]
            self.select_img_type(choice=2)
        except Exception as err:
            logging.exception("Unable to Adjust Brightness/Contrast: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Adjust Brightness/Contrast", 
                                                   "Error trying to adjust image brightness/contrast.Try again."])

    @Slot()
    def apply_microscopy_props_changes(self):
        """Retrieve settings from model and send to Python."""
        try:
            sel_images = self.get_selected_images()
            if len(sel_images) <= 0:
                return
            for val in self.microscopyPropsModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]
                    img.get_pixel_width()
        except Exception as err:
            logging.exception("Unable to Update Microscopy Property Values: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Update Microscopy Values",
                                                   "Error trying to update microscopy property values.Try again."])

    @Slot()
    def apply_img_bin_changes(self):
        """Retrieve settings from model and send to Python."""
        try:
            sel_images = self.get_selected_images()
            if len(sel_images) <= 0:
                return
            for val in self.imgBinFilterModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]
            self.select_img_type()
        except Exception as err:
            logging.exception("Apply Binary Image Filters: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Apply Binary Filters", "Error while tying to apply "
                                                                                     "binary filters to image. Try again."])

    @Slot()
    def apply_img_filter_changes(self):
        """Retrieve settings from model and send to Python."""
        try:
            sel_images = self.get_selected_images()
            if len(sel_images) <= 0:
                return
            for val in self.imgFilterModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]
                    try:
                        img.configs[val["id"]]["dataValue"] = val["dataValue"]
                    except KeyError:
                        pass
            self.select_img_type()
        except Exception as err:
            logging.exception("Apply Image Filters: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Apply Image Filters", "Error while tying to apply "
                                                                                    "image filters. Try again."])

    @Slot()
    def apply_img_scaling(self):
        """Retrieve settings from model and send to Python."""
        try:
            self.set_auto_scale(True)
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.auto_scale = self.allow_auto_scale
            sgt_obj.ntwk_p.scaling_options = self.imgScaleOptionModel.list_data
            sgt_obj.ntwk_p.apply_img_scaling()
            self.select_img_type()
        except Exception as err:
            logging.exception("Apply Image Scaling: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Rescale Image", "Error while tying to re-scale "
                                                                              "image. Try again."])

    @Slot()
    def export_graph(self):
        """Export graph data and save as a file."""
        try:
            sel_images = self.get_selected_images()
            if len(sel_images) <= 0:
                return

            # 1. Get filename
            sgt_obj = self.get_selected_sgt_obj()
            out_dir, filename = sgt_obj.ntwk_p.get_filenames()
            out_dir = out_dir if sgt_obj.ntwk_p.output_dir == '' else sgt_obj.ntwk_p.output_dir

            # 2. Update values
            ntwk_p = sgt_obj.ntwk_p
            for val in self.exportGraphModel.list_data:
                ntwk_p.graph_obj.configs[val["id"]]["value"] = val["value"]

            # 3. Save graph data to file
            ntwk_p.graph_obj.save_graph_to_file(filename, out_dir)
            self.taskTerminatedSignal.emit(True, ["Exporting Graph", "Exported files successfully stored in 'Output Dir'"])
        except Exception as err:
            logging.exception("Unable to Export Graph: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Export Graph", "Error exporting graph to file. Try again."])

    @Slot()
    def save_img_files(self):
        """Retrieve and save images to file."""
        try:

            sel_images = self.get_selected_images()
            if len(sel_images) <= 0:
                return
            for val in self.saveImgModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.save_images_to_file()
            self.taskTerminatedSignal.emit(True,
                                           ["Save Images", "Image files successfully saved in 'Output Dir'"])
        except Exception as err:
            logging.exception("Unable to Save Image Files: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False,
                                           ["Unable to Save Image Files", "Error saving images to file. Try again."])

    @Slot()
    def run_extract_graph(self):
        """Retrieve settings from model and send to Python."""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self.worker_task = WorkerTask()
        try:
            self.wait_flag = True
            sgt_obj = self.get_selected_sgt_obj()

            self.worker = QThreadWorker(func=self.worker_task.task_extract_graph, args=(sgt_obj.ntwk_p,))
            self.worker_task.inProgressSignal.connect(self._handle_progress_update)
            self.worker_task.taskFinishedSignal.connect(self._handle_finished)
            self.worker.start()
        except Exception as err:
            self.wait_flag = False
            logging.exception("Graph Extraction Error: %s", err, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Fatal error occurred! Close the app and try again.")
            self.worker_task.taskFinishedSignal.emit(False, ["Graph Extraction Error",
                                                          "Fatal error while trying to extract graph. "
                                                          "Close the app and try again."])

    @Slot()
    def run_graph_analyzer(self):
        """Retrieve settings from model and send to Python."""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self.worker_task = WorkerTask()
        try:
            self.wait_flag = True
            sgt_obj = self.get_selected_sgt_obj()

            self.worker = QThreadWorker(func=self.worker_task.task_compute_gt, args=(sgt_obj,))
            self.worker_task.inProgressSignal.connect(self._handle_progress_update)
            self.worker_task.taskFinishedSignal.connect(self._handle_finished)
            self.worker.start()
        except Exception as err:
            self.wait_flag = False
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Fatal error occurred! Close the app and try again.")
            self.worker_task.taskFinishedSignal.emit(False, ["GT Computation Error",
                                                          "Fatal error while trying calculate GT parameters. "
                                                          "Close the app and try again."])

    @Slot()
    def run_multi_graph_analyzer(self):
        """"""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self.worker_task = WorkerTask()
        try:
            self.wait_flag = True

            self.worker = QThreadWorker(func=self.worker_task.task_compute_multi_gt, args=(self.sgt_objs,))
            self.worker_task.inProgressSignal.connect(self._handle_progress_update)
            self.worker_task.taskFinishedSignal.connect(self._handle_finished)
            self.worker.start()
        except Exception as err:
            self.wait_flag = False
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self.worker_task.inProgressSignal.emit(-1, "Fatal error occurred! Close the app and try again.")
            self.worker_task.taskFinishedSignal.emit(False, ["GT Computation Error",
                                                          "Fatal error while trying calculate GT parameters. "
                                                          "Close the app and try again."])

    @Slot(result=bool)
    def run_save_project(self):
        """"""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self.wait_flag = True
        success_val = self.save_project_data()
        self.wait_flag = False
        return success_val

    @Slot(result=bool)
    def display_image(self):
        return self.img_loaded

    @Slot(result=bool)
    def image_batches_exist(self):
        if not self.img_loaded:
            return False

        sgt_obj = self.get_selected_sgt_obj()
        batch_count = len(sgt_obj.ntwk_p.image_batches)
        batches_exist = True if batch_count > 1 else False
        return batches_exist

    @Slot(result=bool)
    def is_project_open(self):
        return self.project_open

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
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.crop_image(x, y, width, height)

            # Emit signal to update UI with new image
            self.select_img_type(2)
            self.showCroppingToolSignal.emit(False)
            self.showUnCroppingToolSignal.emit(True)
        except Exception as err:
            logging.exception("Cropping Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Cropping Error", "Error occurred while cropping image. Close the app and try again.")

    @Slot(bool)
    def undo_cropping(self, undo: bool = True):
        if undo:
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.undo_cropping()

            # Emit signal to update UI with new image
            self.select_img_type(None)
            self.showUnCroppingToolSignal.emit(False)

    @Slot(bool)
    def enable_rectangular_selection(self, enabled):
        self.enableRectangularSelectionSignal.emit(enabled)

    @Slot(result=bool)
    def enable_prev_nav_btn(self):
        if (self.selected_sgt_obj_index == 0) or self.is_task_running():
            return False
        else:
            return True

    @Slot(result=bool)
    def enable_next_nav_btn(self):
        if (self.selected_sgt_obj_index == (len(self.sgt_objs) - 1)) or self.is_task_running():
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
            allowed_extensions = tuple(ext[1:] if ext.startswith('*.') else ext for ext in ALLOWED_IMG_EXTENSIONS)
            if a_file.endswith(allowed_extensions):
                img_path = os.path.join(str(img_dir_path), a_file)
                _ = self.create_sgt_object(img_path)

        if len(self.sgt_objs) <= 0:
            logging.info("File Error: Files have to be either .tif .png .jpg .jpeg", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File Error", "No workable images found! Files have to be either .tif, .png, .jpg or .jpeg")
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
            if os.path.exists(proj_path):
                logging.info(f"Project '{proj_name}' already exists.", extra={'user': 'SGT Logs'})
                self.showAlertSignal.emit("Project Error", f"Error: Project '{proj_name}' already exists.")
                return False

            # Open the file in write mode ('w').
            # This will create the file if it doesn't exist
            with open(proj_path, 'w'):
                pass  # Do nothing, just create the file (updates will be done automatically/dynamically)

            # Update and notify QML
            self.project_data["name"] = proj_name
            self.project_data["file_path"] = proj_path
            self.project_open = True
            self.projectOpenedSignal.emit(proj_name)
            logging.info(f"File '{proj_name}' created successfully in '{dir_path}'.", extra={'user': 'SGT Logs'})
        except Exception as err:
            logging.exception("Create Project Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Create Project Error", "Failed to create SGT project. Close the app and try again.")

    @Slot(str, result=bool)
    def open_sgt_project(self, sgt_path):
        """Opens and loads SGT project from the '.sgtproj' file"""
        if self.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return False

        try:
            self.wait_flag = True
            self.project_open = False
            # Verify path
            sgt_path = self.verify_path(sgt_path)
            if not sgt_path:
                self.wait_flag = False
                return False
            img_dir, proj_name = os.path.split(str(sgt_path))

            # Read and load project data and SGT objects
            with open(str(sgt_path), 'rb') as sgt_file:
                self.sgt_objs = pickle.load(sgt_file)

            if self.sgt_objs:
                key_list = list(self.sgt_objs.keys())
                for key in key_list:
                    self.sgt_objs[key].ntwk_p.output_dir = img_dir

            # Update and notify QML
            self.project_data["name"] = proj_name
            self.project_data["file_path"] = str(sgt_path)
            self.wait_flag = False
            self.project_open = True
            self.projectOpenedSignal.emit(proj_name)

            # Load Image to GUI - activates QML
            self.update_img_models(self.get_selected_sgt_obj())
            self.load_image()
            logging.info(f"File '{proj_name}' opened successfully in '{sgt_path}'.", extra={'user': 'SGT Logs'})
            return True
        except Exception as err:
            self.wait_flag = False
            logging.exception("Project Opening Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Open Project Error", "Unable to open .sgtproj file! Try again. If the "
                                                            "issue persists, the file may be corrupted or incompatible. "
                                                            "Consider restoring from a backup or contacting support for "
                                                            "assistance.")
            return False
