# SPDX-License-Identifier: GNU GPL v3
import os
import pickle
import logging
import requests
import numpy as np
from packaging import version
from typing import TYPE_CHECKING, Optional
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Property, Signal, Slot

if TYPE_CHECKING:
    # False at run time, only for a type-checker
    from _typeshed import SupportsWrite

from .tree_model import TreeModel
from .table_model import TableModel
from .checkbox_model import CheckBoxModel
from .imagegrid_model import ImageGridModel
from .qthread_worker import QThreadWorker
from ..base_worker import BaseWorker
from ..base_contoller import BaseController

from ... import __version__, __title__
from ...utils.sgt_utils import img_to_base64, verify_path
from ...imaging.image_processor import ImageProcessor, FiberNetworkBuilder, ALLOWED_IMG_EXTENSIONS, ALLOWED_GRAPH_FILE_EXTENSIONS
from ...compute.graph_analyzer import GraphAnalyzer#, COMPUTING_DEVICE


class MainController(BaseController):
    """Exposes a method to refresh the image in QML"""

    _waitFlagChanged = Signal()
    errorSignal = Signal(str)
    showAlertSignal = Signal(str, str)
    updateProgressSignal = Signal(int, str)
    taskTerminatedSignal = Signal(bool, list)
    projectOpenedSignal = Signal(str)
    changeImageSignal = Signal()
    imageChangedSignal = Signal()
    showImageHistogramSignal = Signal(bool)
    enableRectangularSelectionSignal = Signal(bool)
    showCroppingToolSignal = Signal(bool)
    showUnCroppingToolSignal = Signal(bool)
    performCroppingSignal = Signal(bool)

    def __init__(self, qml_app: QApplication):
        super().__init__()
        self._qml_app = qml_app
        self._img_loaded = False
        self._project_open = False
        self._applying_changes = False

        # Project data
        self._project_data = {"name": "", "file_path": ""}
        self._software_update = "No updates available!"

        # Initialize flags
        self._wait_flag, self._wait_flag_hist = False, False
        self._wait_msg = ""

        # Create Models
        self.imgThumbnailModel = TableModel([])
        self.imagePropsModel = TableModel([])
        self.graphPropsModel = TableModel([])
        self.graphComputeModel = TableModel([])
        self.microscopyPropsModel = CheckBoxModel([])
        self.gtcScalingModel = CheckBoxModel([])

        self.gteTreeModel = TreeModel([])
        self.gtcListModel = CheckBoxModel([])
        self.exportGraphModel = CheckBoxModel([])
        self.imgBatchModel = CheckBoxModel([])
        self.imgControlModel = CheckBoxModel([])
        self.imgBinFilterModel = CheckBoxModel([])
        self.imgFilterModel = CheckBoxModel([])
        self.imgScaleOptionModel = CheckBoxModel([])
        self.imgViewOptionModel = CheckBoxModel([])
        self.saveImgModel = CheckBoxModel([])
        self.img3dGridModel = ImageGridModel([], set([]))
        self.imgHistogramModel = ImageGridModel([], set([]))

        # Create QThreadWorker for long tasks
        self._thread_worker, self._task_worker = QThreadWorker(0, None), BaseWorker()
        self._thread_worker_hist, self._hist_worker = QThreadWorker(0, None), BaseWorker()

    def synchronize_img_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload image configuration selections and controls from saved dict to QML gui_mcw after the image is loaded.

            :param sgt_obj: A GraphAnalyzer object with all saved user-selected configurations.
        """
        try:
            # Models Auto-update with saved sgt_obj configs. No need to re-assign!
            ntwk_p = sgt_obj.ntwk_p
            sel_img_batch = ntwk_p.selected_batch
            options_img = ntwk_p.image_obj.configs

            img_controls = [v for v in options_img.values() if v["type"] == "image-control"]
            bin_filters = [v for v in options_img.values() if v["type"] == "binary-filter"]
            img_filters = [v for v in options_img.values() if v["type"] == "image-filter"]
            img_properties = [v for v in options_img.values() if v["type"] == "image-property"]
            file_options = [v for v in options_img.values() if v["type"] == "file-options"]

            batch_list = [{"id": f"batch_{i}", "text": f"Image Batch {i+1}", "value": i}
                          for i in range(len(sgt_obj.ntwk_p.image_batches))]
            self.imgBatchModel.reset_data(batch_list)
            self.imgScaleOptionModel.reset_data(sel_img_batch.scaling_options)
            self.imgViewOptionModel.reset_data(sel_img_batch.view_options)

            self.imgControlModel.reset_data(img_controls)
            self.imgBinFilterModel.reset_data(bin_filters)
            self.imgFilterModel.reset_data(img_filters)
            self.microscopyPropsModel.reset_data(img_properties)
            self.saveImgModel.reset_data(file_options)
        except Exception as err:
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Fatal Error", "Error re-loading image configurations! Close app and try again.")

    def synchronize_graph_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload graph configuration selections and controls from saved dict to QML gui_mcw.
        Args:
            sgt_obj: a GraphAnalyzer object with all saved user-selected configurations.

        Returns:

        """
        try:
            # Models Auto-update with saved sgt_obj configs. No need to re-assign!
            ntwk_p = sgt_obj.ntwk_p
            sel_img_batch = ntwk_p.selected_batch
            graph_obj = ntwk_p.graph_obj
            option_gte = graph_obj.configs
            options_gtc = sgt_obj.configs

            graph_options = [v for v in option_gte.values() if v["type"] == "graph-extraction"]
            file_options = [v for v in option_gte.values() if v["type"] == "file-options"]
            compute_options = [v for v in options_gtc.values() if v["type"] == "gt-metric"]
            scaling_options = [v for v in options_gtc.values() if v["type"] == "scaling-param"]

            self.gteTreeModel.reset_data(graph_options)
            self.exportGraphModel.reset_data(file_options)
            self.gtcListModel.reset_data(compute_options)
            self.gtcScalingModel.reset_data(scaling_options)

            self.imagePropsModel.reset_data(sel_img_batch.props)
            self.graphPropsModel.reset_data(graph_obj.props)
            self.graphComputeModel.reset_data(sgt_obj.props)
        except Exception as err:
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Fatal Error", "Error re-loading image configurations! Close app and try again.")

    def delete_sgt_object(self, index=None):
        """
        Delete SGT Obj stored at the specified index (if not specified, get the current index).
        """
        del_index = index if index is not None else self._selected_sgt_obj_index
        if 0 <= del_index < len(self._sgt_objs):  # Check if the index exists
            keys_list = list(self._sgt_objs.keys())
            key_at_del_index = keys_list[self._selected_sgt_obj_index]
            # Delete the object at index
            del self._sgt_objs[key_at_del_index]
            # Update Data
            img_list, img_cache = self.get_thumbnail_list()
            self.imgThumbnailModel.update_data(img_list, img_cache)
            self.imagePropsModel.reset_data([])
            self.graphPropsModel.reset_data([])
            self.graphComputeModel.reset_data([])
            self._selected_sgt_obj_index = 0
            self.load_image(reload_thumbnails=True)
            self.imageChangedSignal.emit()

    def save_project_data(self):
        """
        A handler function that handles saving project data.
        Returns: True if successful, False otherwise.

        """
        if not self._project_open:
            return False
        try:
            file_path = self._project_data["file_path"]
            with open(file_path, 'wb') as project_file:  # type: Optional[SupportsWrite[bytes]]
                pickle.dump(self._sgt_objs, project_file)
            return True
        except Exception as err:
            logging.exception("Project Saving Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Save Error", "Unable to save project data. Close app and try again.")
            return False

    def get_thumbnail_list(self):
        """
        Get names and base64 data of images to be used in Project List thumbnails.
        """
        keys_list = list(self._sgt_objs.keys())
        if len(keys_list) <= 0:
            return None, None
        item_data = []
        image_cache = {}
        for key in keys_list:
            item_data.append([key])  # Store the key
            sgt_obj = self._sgt_objs[key]
            if sgt_obj.ntwk_p.selected_batch.is_graph_only:
                empty_cv = np.ones((256, 256), dtype=np.uint8) * 255
                img_cv = empty_cv if sgt_obj.ntwk_p.graph_obj.img_ntwk is None else sgt_obj.ntwk_p.graph_obj.img_ntwk
            else:
                img_cv = sgt_obj.ntwk_p.image_2d
            base64_data = img_to_base64(img_cv)
            image_cache[key] = base64_data  # Store base64 string
        return item_data, image_cache

    def get_selected_images(self):
        """
        Get selected images from a specific image batch.
        """
        sgt_obj = self.get_selected_sgt_obj()
        ntwk_p = sgt_obj.ntwk_p
        return ntwk_p.selected_images

    def _handle_progress_update(self, progress_val: int, msg: str) -> None:
        """
        Handler function for progress updates for ongoing tasks.
        Args:
            progress_val: Progress value, range is 0-100%.
            msg: Progress message to be displayed.

        Returns:

        """

        if 0 <= progress_val <= 100:
            self.updateProgressSignal.emit(progress_val, msg)
            logging.info(f"{progress_val}%: {msg}", extra={'user': 'SGT Logs'})
        elif progress_val > 100:
            self.updateProgressSignal.emit(progress_val, msg)
            logging.info(f"{msg}", extra={'user': 'SGT Logs'})
        else:
            logging.exception("Error: %s", msg, extra={'user': 'SGT Logs'})
            self.errorSignal.emit(msg)

    def _handle_finished(self, success_val: bool, result: None | list | ImageProcessor | FiberNetworkBuilder | GraphAnalyzer) -> None:
        """
        Handler function for sending updates/signals on termination of tasks.
        Args:
            success_val:
            result:

        Returns:

        """
        self._stop_wait()
        if not success_val:
            if type(result) is list:
                logging.info(result[0] + ": " + result[1], extra={'user': 'SGT Logs'})
                self.taskTerminatedSignal.emit(success_val, result)
            # elif type(result) is GraphAnalyzer:
            #    pdf_saved = GraphAnalyzer.write_to_pdf(result, self._handle_progress_update)
            #    if pdf_saved:
            #        self._handle_finished(True, result)
        else:
            if type(result) is str:
                # Saving files to Output Folder
                self._handle_progress_update(100, "Files Saved!")
                self.taskTerminatedSignal.emit(success_val, ["Files Saved", result])
            elif type(result) is ImageProcessor:
                self._handle_progress_update(100, "Graph extracted successfully!")
                # Update Compute properties
                self.changeImageSignal.emit()
                self.synchronize_graph_models(self.get_selected_sgt_obj())
                # Send task termination signal to QML
                self.taskTerminatedSignal.emit(success_val, [])
            elif type(result) is GraphAnalyzer:
                self._handle_progress_update(100, "GT PDF successfully generated!")
                # Update Compute properties
                self.synchronize_graph_models(self.get_selected_sgt_obj())
                # Send task termination signal to QML
                self.taskTerminatedSignal.emit(True, ["GT calculations completed", "The image's GT parameters have been "
                                                                                "calculated. Check out generated PDF in "
                                                                                "'Output Dir'."])
            elif type(result) is dict:
                self._handle_progress_update(100, "All GT PDF successfully generated!")
                # Update Compute properties
                self.synchronize_graph_models(self.get_selected_sgt_obj())
                # Send task termination signal to QML
                self.taskTerminatedSignal.emit(True, ["All GT calculations completed", "GT parameters of all "
                                                                                    "images have been calculated. Check "
                                                                                    "out all the generated PDFs in "
                                                                                    "'Output Dir'."])
            elif type(result) is list:
                # Image histogram calculated
                self._wait_flag_hist = False
                if len(self._sgt_objs) > 0:
                    sgt_obj = self.get_selected_sgt_obj()
                    sel_img_batch = sgt_obj.ntwk_p.selected_batch
                    self.imgHistogramModel.reset_data(result, sel_img_batch.selected_images_idx)
            else:
                self.taskTerminatedSignal.emit(success_val, [])

            # Auto-save changes to the project data file
            if len(self._sgt_objs.items()) <= 10:
                self.save_project_data()

    def _start_wait(self, msg: str = "please wait..."):
        """Activate the wait flag and send a wait signal."""
        self._wait_flag = True
        self._wait_msg = msg
        self._waitFlagChanged.emit()

    def _stop_wait(self):
        """Deactivate the wait flag and send a wait signal."""
        self._wait_flag = False
        self._wait_msg = ""
        self._waitFlagChanged.emit()

    # --- Properties exposed to QML because of "notify" ---
    @Property(bool, notify=_waitFlagChanged)
    def wait(self):
        return self._wait_flag

    @Property(str, notify=_waitFlagChanged)
    def wait_text(self):
        return self._wait_msg

    @Slot(result=str)
    def get_sgt_title(self):
        return f"{__title__}"

    @Slot(result=str)
    def get_sgt_version(self):
        """"""
        # return f"{__title__} v{__version__}, Computing: {COMPUTING_DEVICE}"
        return f"v{__version__}"

    @Slot(result=str)
    def get_software_download_details(self):
        return self._software_update

    @Slot(result=str)
    def get_about_details(self):
        about_app = (
            "<html>"
            "<p>"
            "A software tool for performing Graph Theory analysis on <br>microscopy images. This is a modified version "
            "of StructuralGT <br>initially proposed by Drew A. Vecchio,<br>"
            "<b>DOI:</b> <a href='https://pubs.acs.org/doi/10.1021/acsnano.1c04711'>10.1021/acsnano.1c04711</a>"
            "</p><p>"
            "<b>Main Contributors:</b><br>"
            "<table border='0.5' cellspacing='0' cellpadding='4'>"
            # "<tr><th>Name</th><th>Email</th></tr>"
            "<tr><td>Dickson Owuor</td><td>owuor@umich.edu</td></tr>"
            "<tr><td>Nicolas Kotov</td><td>kotov@umich.edu</td></tr>"
            "<tr><td>Alain Kadar</td><td>alaink@umich.edu</td></tr>"
            "<tr><td>Xiong Ye Xiao</td><td>xiongyex@usc.edu</td></tr>"
            "<tr><td>Kotov Lab</td><td></td></tr>"
            "<tr><td>COMPASS</td><td></td></tr>"
            "</table></p><p><br><br>"
            "<b>Documentation:</b> <a href='https://structural-gt.readthedocs.io'>structural-gt.readthedocs.io</a>"
            "<br>"
            f"<b> Version: </b> {self.get_sgt_version()}<br>"
            "<b>License:</b> GPL GNU v3"
            "</p><p>"
            "Copyright (C) 2018-2025<br>The Regents of the University of Michigan."
            "</p>"
            "</html>")
        return about_app

    @Slot(result=bool)
    def check_for_updates(self):
        """"""
        github_url = "https://raw.githubusercontent.com/owuordickson/structural-gt/refs/heads/main/src/sgtlib/__init__.py"

        try:
            response = requests.get(github_url, timeout=5)
            response.raise_for_status()
        except requests.RequestException as e:
            self._software_update = f"Error checking for updates: {e}"
            return False

        remote_version = None
        for line in response.text.splitlines():
            if line.strip().startswith("__install_version__"):
                try:
                    remote_version = line.split("=")[1].strip().strip("\"'")
                    break
                except IndexError:
                    self._software_update = "Could not connect to server!"
                    return False

        if not remote_version:
            self._software_update = "Could not find the new version!"
            return False

        new_version = version.parse(remote_version)
        current_version = version.parse(__version__)
        if new_version > current_version:
            # https://github.com/owuordickson/structural-gt/releases/tag/v3.3.5
            self._software_update = (
                "New version available!<br>"
                f"Download via this <a href='https://github.com/owuordickson/structural-gt/releases/tag/v{remote_version}'>link</a>"
            )
            return True
        else:
            self._software_update = "No updates available."
            return False

    @Slot(str, result=str)
    def get_file_extensions(self, option):
        if option == "img":
            pattern_string = ' '.join(ALLOWED_IMG_EXTENSIONS)
            return f"Image files ({pattern_string})"
        if option == "graph":
            pattern_string = ' '.join(ALLOWED_GRAPH_FILE_EXTENSIONS)
            return f"Graph files ({pattern_string})"
        elif option == "proj":
            return "Project files (*.sgtproj)"
        else:
            return ""

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        curr_img_view = np.random.randint(0, 4)
        unique_num = self._selected_sgt_obj_index + curr_img_view + np.random.randint(low=21, high=1000)
        return "image://imageProvider/" + str(unique_num)

    @Slot(result=bool)
    def is_img_3d(self):
        sgt_obj = self.get_selected_sgt_obj()
        if sgt_obj is None:
            return False
        sel_img_batch = sgt_obj.ntwk_p.selected_batch
        is_3d = not sel_img_batch.is_2d
        return is_3d

    @Slot(result=int)
    def get_selected_img_batch(self):
        try:
            sgt_obj = self.get_selected_sgt_obj()
            return sgt_obj.ntwk_p.selected_batch_index
        except AttributeError:
            logging.exception("No image added! Please add at least one image.", extra={'user': 'SGT Logs'})
            return 0

    @Slot(result=str)
    def get_img_nav_location(self):
        return f"{(self._selected_sgt_obj_index + 1)} / {len(self._sgt_objs)}"

    @Slot(result=str)
    def get_output_dir(self):
        sgt_obj = self.get_selected_sgt_obj()
        if sgt_obj is None:
            return ""
        return f"{sgt_obj.ntwk_p.output_dir}"

    @Slot(result=bool)
    def get_auto_scale(self):
        return self._allow_auto_scale

    @Slot(int)
    def delete_selected_thumbnail(self, img_index):
        """Delete the selected image from the list."""
        self.delete_sgt_object(img_index)

    @Slot(str)
    def set_output_dir(self, folder_path):
        self.update_output_dir(folder_path)
        self.imageChangedSignal.emit()

    @Slot(bool)
    def set_auto_scale(self, auto_scale):
        """Set the auto-scale parameter for each image."""
        self._allow_auto_scale = auto_scale

    @Slot(int)
    def select_img_batch(self, batch_index=-1):
        if batch_index < 0:
            return

        try:
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.select_image_batch(batch_index)
            # Load the SGT Object data of the selected image
            self.synchronize_img_models(sgt_obj)
            self.synchronize_graph_models(self.get_selected_sgt_obj())
            # Load the selected image into the view
            self.changeImageSignal.emit()
        except Exception as err:
            logging.exception("Batch Change Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Image Batch Error", f"Error encountered while trying to access batch "
                                                           f"{batch_index}. Restart app and try again.")

    @Slot(int, bool)
    def toggle_selected_batch_image(self, img_index, selected):
        sgt_obj = self.get_selected_sgt_obj()
        sel_img_batch = sgt_obj.ntwk_p.selected_batch
        if selected:
            sel_img_batch.selected_images_idx.add(img_index)
        else:
            sel_img_batch.selected_images_idx.discard(img_index)
        self.changeImageSignal.emit()

    @Slot(bool)
    def reload_graph_image(self, only_giant_graph=False):
        sgt_obj = self.get_selected_sgt_obj()
        sel_img_batch = sgt_obj.ntwk_p.selected_batch
        sgt_obj.ntwk_p.draw_graph_image(sel_img_batch, show_giant_only=only_giant_graph)
        self.changeImageSignal.emit()

    @Slot()
    def load_graph_simulation(self):
        """Render and visualize OVITO graph network simulation."""
        try:
            # Import libraries
            from ovito import scene
            from ovito.vis import Viewport
            from ovito.io import import_file
            from ovito.gui import create_qwidget

            # Clear any existing scene
            for p_line in list(scene.pipelines):
                p_line.remove_from_scene()

            # Create OVITO data pipeline
            sgt_obj = self.get_selected_sgt_obj()
            h, w = sgt_obj.ntwk_p.graph_obj.img_ntwk.shape[:2]
            pipeline = import_file(sgt_obj.ntwk_p.graph_obj.gsd_file)
            pipeline.add_to_scene()

            vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(2, 1, -1))
            vp.zoom_all((w, h))  # width, height

            ovito_widget = create_qwidget(vp, parent=self._qml_app.activeWindow())
            ovito_widget.setFixedSize(w, h)
            ovito_widget.show()
        except Exception as e:
            print("Graph Simulation Error:", e)

    @Slot(int)
    def load_image(self, index=None, reload_thumbnails=False):
        try:
            if index is not None:
                if index == self._selected_sgt_obj_index:
                    return
                else:
                    self._selected_sgt_obj_index = index

            if reload_thumbnails:
                # Update the thumbnail list data (delete/add image)
                img_list, img_cache = self.get_thumbnail_list()
                self.imgThumbnailModel.update_data(img_list, img_cache)

            # Load the SGT Object data of the selected image
            self.synchronize_img_models(self.get_selected_sgt_obj())
            self.synchronize_graph_models(self.get_selected_sgt_obj())
            self.imgThumbnailModel.set_selected(self._selected_sgt_obj_index)
            # Load the selected image into the view
            self.changeImageSignal.emit()
        except Exception as err:
            self.delete_sgt_object()
            self._selected_sgt_obj_index = 0
            logging.exception("Image Loading Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Image Error", "Error loading image. Try again.")

    @Slot(result=bool)
    def load_prev_image(self):
        """Load the previous image in the list into view."""
        if self._selected_sgt_obj_index > 0:
            prev_img_idx = self._selected_sgt_obj_index - 1
            self.load_image(index=prev_img_idx)
            return True
        return False

    @Slot(result=bool)
    def load_next_image(self):
        """Load the next image in the list into view."""
        if self._selected_sgt_obj_index < (len(self._sgt_objs) - 1):
            next_img_idx = self._selected_sgt_obj_index + 1
            self.load_image(index=next_img_idx)
            return True
        return False

    @Slot(str)
    def apply_changes(self, view: str=""):
        """Retrieve changes made by the user and apply to image/graph."""
        if not self._applying_changes:  # Disallow concurrent changes
            # print(f"change to {view}")
            self._applying_changes = True
            if view != "":
                sgt_obj = self.get_selected_sgt_obj()
                sgt_obj.ntwk_p.selected_batch_view = view
            self.changeImageSignal.emit()

    @Slot()
    def compute_img_histogram(self):
        """Calculate the histogram of the image."""
        if self._wait_flag_hist:
            return

        self.showImageHistogramSignal.emit(True)
        self._hist_worker = BaseWorker()
        try:
            self._wait_flag_hist = True
            sgt_obj = self.get_selected_sgt_obj()
            self._thread_worker_hist = QThreadWorker(func=self._hist_worker.task_calculate_img_histogram, args=(sgt_obj.ntwk_p,))
            self._hist_worker.taskFinishedSignal.connect(self._handle_finished)
            self._thread_worker_hist.start()
        except Exception as err:
            self._wait_flag_hist = False
            logging.exception("Histogram Calculation Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_finished(False, ["Histogram Calculation Failed", "Unable to calculate image histogram!"])

    @Slot()
    def apply_img_scaling(self):
        """Retrieve settings from the model and send to Python."""
        try:
            self.set_auto_scale(True)
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.auto_scale = self._allow_auto_scale
            sgt_obj.ntwk_p.apply_img_scaling()
            self.changeImageSignal.emit()
        except Exception as err:
            logging.exception("Apply Image Scaling: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Rescale Image", "Error while tying to re-scale "
                                                                              "image. Try again."])

    @Slot()
    def export_graph_to_file(self):
        """Export graph data and save as a file."""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self._handle_progress_update(0, "Exporting Graph Data...")
        self._task_worker = BaseWorker()
        try:
            if self.get_selected_sgt_obj().ntwk_p.selected_batch.is_graph_only:
                return

            self._handle_progress_update(20, "Exporting Graph Data...")
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()

            self._thread_worker = QThreadWorker(func=self._task_worker.task_export_graph, args=(sgt_obj.ntwk_p,))
            self._task_worker.taskFinishedSignal.connect(self._handle_finished)
            self._thread_worker.start()
        except Exception as err:
            logging.exception("Unable to Export Graph: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False, ["Unable to Export Graph", "Error exporting graph to file. Try again."])

    @Slot()
    def save_img_files(self):
        """Retrieve and save images to the file."""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self._handle_progress_update(0, "Saving Images...")
        self._task_worker = BaseWorker()
        try:
            if self.get_selected_sgt_obj().ntwk_p.selected_batch.is_graph_only:
                return

            self._handle_progress_update(10, "Saving Images...")
            sel_images = self.get_selected_images()
            for val in self.saveImgModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]

            self._handle_progress_update(20, "Saving Images...")
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()

            self._thread_worker = QThreadWorker(func=self._task_worker.task_save_images, args=(sgt_obj.ntwk_p,))
            self._task_worker.taskFinishedSignal.connect(self._handle_finished)
            self._thread_worker.start()
        except Exception as err:
            logging.exception("Unable to Save Image Files: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False,
                                           ["Unable to Save Image Files", "Error saving images to file. Try again."])

    @Slot()
    def run_extract_graph(self):
        """Retrieve settings from the model and send to Python."""

        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self._task_worker = BaseWorker()
        try:
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()

            self._thread_worker = QThreadWorker(func=self._task_worker.task_extract_graph, args=(sgt_obj.ntwk_p,))
            self._task_worker.inProgressSignal.connect(self._handle_progress_update)
            self._task_worker.taskFinishedSignal.connect(self._handle_finished)
            self._thread_worker.start()
        except Exception as err:
            self._stop_wait()
            logging.exception("Graph Extraction Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_progress_update(-1, "Fatal error occurred! Close the app and try again.")
            self._handle_finished(False, ["Graph Extraction Error",
                                                          "Fatal error while trying to extract graph. "
                                                          "Close the app and try again."])

    @Slot()
    def run_graph_analyzer(self):
        """Retrieve settings from the model and send to Python."""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self._task_worker = BaseWorker()
        try:
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()

            self._thread_worker = QThreadWorker(func=self._task_worker.task_compute_gt, args=(sgt_obj,))
            self._task_worker.inProgressSignal.connect(self._handle_progress_update)
            self._task_worker.taskFinishedSignal.connect(self._handle_finished)
            self._thread_worker.start()
        except Exception as err:
            self._stop_wait()
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_progress_update(-1, "Fatal error occurred! Close the app and try again.")
            self._handle_finished(False, ["GT Computation Error",
                                                          "Fatal error while trying calculate GT parameters. "
                                                          "Close the app and try again."])

    @Slot()
    def run_multi_graph_analyzer(self):
        """"""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self._task_worker = BaseWorker()
        try:
            self._start_wait()

            # Update Configs
            self.replicate_sgt_configs()

            self._thread_worker = QThreadWorker(func=self._task_worker.task_compute_multi_gt, args=(self._sgt_objs,))
            self._task_worker.inProgressSignal.connect(self._handle_progress_update)
            self._task_worker.taskFinishedSignal.connect(self._handle_finished)
            self._thread_worker.start()
        except Exception as err:
            self._stop_wait()
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_progress_update(-1, "Fatal error occurred! Close the app and try again.")
            self._handle_finished(False, ["GT Computation Error",
                                                          "Fatal error while trying calculate GT parameters. "
                                                          "Close the app and try again."])

    @Slot(result=bool)
    def run_save_project(self):
        """"""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return False

        self._start_wait()
        success_val = self.save_project_data()
        self._stop_wait()
        return success_val

    @Slot(result=bool)
    def enable_img_controls(self):
        """Enable image controls."""
        if len(self._sgt_objs) <= 0:
            return False

        sgt_obj = self.get_selected_sgt_obj()
        if sgt_obj is None:
            return False
        return not sgt_obj.ntwk_p.selected_batch.is_graph_only

    @Slot(result=bool)
    def display_image(self):
        return self._img_loaded

    @Slot(result=bool)
    def display_graph(self):
        if len(self._sgt_objs) <= 0:
            return False

        sgt_obj = self.get_selected_sgt_obj()
        if sgt_obj is None:
            return False

        if sgt_obj.ntwk_p.graph_obj.img_ntwk is None:
            return False

        if sgt_obj.ntwk_p.selected_batch_view == "graph":
            return True
        return False

    @Slot(result=bool)
    def image_batches_exist(self):
        if not self._img_loaded:
            return False

        sgt_obj = self.get_selected_sgt_obj()
        batch_count = len(sgt_obj.ntwk_p.image_batches)
        batches_exist = True if batch_count > 1 else False
        return batches_exist

    @Slot(result=bool)
    def is_project_open(self):
        return self._project_open

    @Slot(result=bool)
    def is_task_running(self):
        return self._wait_flag

    @Slot(bool)
    def show_cropping_tool(self, allow_cropping):
        self.showCroppingToolSignal.emit(allow_cropping)

    @Slot(bool)
    def perform_cropping(self, allowed):
        self.performCroppingSignal.emit(allowed)

    @Slot( int, int, int, int, int, int)
    def crop_image(self, x, y, crop_width, crop_height, qimg_width, qimg_height):
        """Crop image using PIL and save it."""
        try:
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.crop_image(x, y, crop_width, crop_height, qimg_width, qimg_height)

            # Emit signal to update UI with new image
            self.changeImageSignal.emit()
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
            self.changeImageSignal.emit()
            self.showUnCroppingToolSignal.emit(False)

    @Slot(bool)
    def enable_rectangular_selection(self, enabled):
        self.enableRectangularSelectionSignal.emit(enabled)

    @Slot(result=bool)
    def enable_prev_nav_btn(self):
        if (self._selected_sgt_obj_index == 0) or self.is_task_running():
            return False
        else:
            return True

    @Slot(result=bool)
    def enable_next_nav_btn(self):
        if (self._selected_sgt_obj_index == (len(self._sgt_objs) - 1)) or self.is_task_running():
            return False
        else:
            return True

    @Slot(str, result=bool)
    def upload_graph_file(self, file_path):
        """Verify and validate the file path, use it to create a new SGT Object and load it into the view."""
        is_successful = self.add_graph(file_path)
        if is_successful:
            self.synchronize_img_models(self.get_selected_sgt_obj())
            self.synchronize_graph_models(self.get_selected_sgt_obj())
            self.load_image(reload_thumbnails=True)
        return is_successful

    @Slot(str, result=bool)
    def upload_single_image(self, img_path):
        """Verify and validate the image path, use it to create an SGT object and load it in view."""
        is_successful = self.add_single_image(img_path)
        if is_successful:
            self.synchronize_img_models(self.get_selected_sgt_obj())
            self.synchronize_graph_models(self.get_selected_sgt_obj())
            self.load_image(reload_thumbnails=True)
        return is_successful

    @Slot(str, result=bool)
    def upload_multiple_images(self, img_dir_path):
        """
        Verify and validate multiple image paths, use each to create an SGT object, then load the last one in view.
        """
        is_successful = self.add_multiple_images(img_dir_path)
        if is_successful:
            self.synchronize_img_models(self.get_selected_sgt_obj())
            self.synchronize_graph_models(self.get_selected_sgt_obj())
            self.load_image(reload_thumbnails=True)
        return is_successful

    @Slot(str, str, result=bool)
    def create_sgt_project(self, proj_name, dir_path):
        """Creates a '.sgtproj' inside the selected directory"""

        self._project_open = False
        success, result = verify_path(dir_path)
        if success:
            dir_path = result
        else:
            logging.info(result, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File/Directory Error", result)
            return False

        proj_name += '.sgtproj'
        proj_path = os.path.join(str(dir_path), proj_name)

        try:
            if os.path.exists(proj_path):
                logging.info(f"Project '{proj_name}' already exists.", extra={'user': 'SGT Logs'})
                self.showAlertSignal.emit("Project Error", f"Error: Project '{proj_name}' already exists.")
                return False

            # Open the file in the 'write' mode ('w').
            # This will create the file if it doesn't exist
            with open(proj_path, 'w'):
                pass  # Do nothing, just create the file (updates will be done automatically/dynamically)

            # Update and notify QML
            self._project_data["name"] = proj_name
            self._project_data["file_path"] = proj_path
            self._project_open = True
            self.projectOpenedSignal.emit(proj_name)
            logging.info(f"File '{proj_name}' created successfully in '{dir_path}'.", extra={'user': 'SGT Logs'})
            return True
        except Exception as err:
            logging.exception("Create Project Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Create Project Error", "Failed to create SGT project. Close the app and try again.")
            return False

    @Slot(str, result=bool)
    def open_sgt_project(self, sgt_path):
        """Opens and loads the SGT project from the '.sgtproj' file"""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return False

        try:
            self._start_wait()
            self._project_open = False
            # Verify the path
            success, result = verify_path(sgt_path)
            if success:
                sgt_path = result
            else:
                logging.info(result, extra={'user': 'SGT Logs'})
                self.showAlertSignal.emit("File/Directory Error", result)
                self._stop_wait()
                return False
            img_dir, proj_name = os.path.split(str(sgt_path))

            # Read and load project data and SGT objects
            with open(str(sgt_path), 'rb') as sgt_file:
                self._sgt_objs = pickle.load(sgt_file)

            if self._sgt_objs:
                key_list = list(self._sgt_objs.keys())
                for key in key_list:
                    self._sgt_objs[key].ntwk_p.output_dir = img_dir

            # Update and notify QML
            self._project_data["name"] = proj_name
            self._project_data["file_path"] = str(sgt_path)
            self._stop_wait()
            self._project_open = True
            self.projectOpenedSignal.emit(proj_name)

            # Load Image to GUI - activates QML
            self.load_image(reload_thumbnails=True)
            logging.info(f"File '{proj_name}' opened successfully in '{sgt_path}'.", extra={'user': 'SGT Logs'})
            return True
        except Exception as err:
            self._stop_wait()
            logging.exception("Project Opening Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Open Project Error", "Unable to open .sgtproj file! Try again. If the "
                                                            "issue persists, the file may be corrupted or incompatible. "
                                                            "Consider restoring from a backup or contacting support for "
                                                            "assistance.")
            return False
