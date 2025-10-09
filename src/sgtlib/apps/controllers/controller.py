# SPDX-License-Identifier: GNU GPL v3
import os
import pickle
import logging
import requests
import numpy as np
from packaging import version
from typing import TYPE_CHECKING, Optional
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Signal, Slot

if TYPE_CHECKING:
    # False at run time, only for a type-checker
    from _typeshed import SupportsWrite

from ..models.tree_model import TreeModel
from ..models.table_model import TableModel
from ..models.checkbox_model import CheckBoxModel
from ..models.imagegrid_model import ImageGridModel
from ..workers.persistent_worker import PersistentProcessWorker
from ..workers.base_workers import BaseWorker
from .base_contoller import BaseController

from ... import __version__, __title__
from ...utils.sgt_utils import img_to_base64, verify_path, TaskResult, ProgressData
from ...imaging.image_processor import ALLOWED_IMG_EXTENSIONS, ALLOWED_GRAPH_FILE_EXTENSIONS
from ...compute.graph_analyzer import GraphAnalyzer  # , COMPUTING_DEVICE


class MainController(BaseController):
    """Exposes a method to refresh the image in QML"""

    errorSignal = Signal(str)
    updateProgressSignal = Signal(int, str)
    updateAIProgressSignal = Signal(int, str)
    taskTerminatedSignal = Signal(bool, list)
    projectOpenedSignal = Signal(str)
    changeImageSignal = Signal()
    imageChangedSignal = Signal()
    showImageFilterControls = Signal(bool)
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
        self.imgColorsModel = CheckBoxModel([])
        self.aiSearchModel = CheckBoxModel([])

        self.imgScaleOptionModel = CheckBoxModel([])
        self.imgViewOptionModel = CheckBoxModel([])
        self.saveImgModel = CheckBoxModel([])
        self.img3dGridModel = ImageGridModel([], set([]))
        self.imgHistogramModel = ImageGridModel([], set([]))

        # Create Persistent Workers (Processes) - better than threads in handling long tasks (not affected by GIL)
        self._gt_worker = PersistentProcessWorker(worker_id=1)
        self._ai_worker = PersistentProcessWorker(worker_id=2)
        self._hist_worker = PersistentProcessWorker(worker_id=3)

    def synchronize_img_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload image configuration selections and controls from saved dict to QML gui_mcw after the image is loaded.

            :param sgt_obj: A GraphAnalyzer object with all saved user-selected configurations.
        """
        try:
            # Models Auto-update with saved sgt_obj configs. No need to re-assign!
            ntwk_p = sgt_obj.ntwk_p
            sel_img_batch = ntwk_p.selected_batch
            options_ai = ntwk_p.configs
            options_img = ntwk_p.image_obj.configs

            # Get data from object configs
            img_controls = [v for v in options_img.values() if v["type"] == "image-control"]
            bin_filters = [v for v in options_img.values() if v["type"] == "binary-filter"]
            img_filters = [v for v in options_img.values() if v["type"] == "image-filter"]
            img_properties = [v for v in options_img.values() if v["type"] == "image-property"]
            file_options = [v for v in options_img.values() if v["type"] == "file-options"]
            ai_search_params = [v for v in options_ai.values() if v["type"] == "search-params"]

            batch_list = [{"id": f"batch_{i}", "text": f" Batch {i + 1}", "value": i}
                          for i in range(len(sgt_obj.ntwk_p.image_batches))]

            # Update QML adapter-models with fetched data
            self.imgBatchModel.reset_data(batch_list)
            self.imgScaleOptionModel.reset_data(sel_img_batch.scaling_options)
            self.imgViewOptionModel.reset_data(sel_img_batch.view_options)

            self.imgControlModel.reset_data(img_controls)
            self.imgBinFilterModel.reset_data(bin_filters)
            self.imgFilterModel.reset_data(img_filters)
            self.aiSearchModel.reset_data(ai_search_params)
            self.saveImgModel.reset_data(file_options)

            self.microscopyPropsModel.reset_data(img_properties)
            self.imagePropsModel.reset_data(sel_img_batch.props)
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

            self.graphPropsModel.reset_data(graph_obj.props)
            self.graphComputeModel.reset_data(sgt_obj.props)
        except Exception as err:
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Fatal Error", "Error re-loading image configurations! Close app and try again.")

    def reset_qml_models(self, only_colors: bool = False):
        """
        Reset some of the CheckBox models when different image is loaded.

        Args:
            only_colors: If True, reset only the imgColorsModel model. If False, reset all models.

        Returns:
            None
        """
        print("Resetting QML models...")
        # Erase existing data in QML adapter-models
        self.imgColorsModel.reset_data([])
        if only_colors:
            return
        self.imgHistogramModel.reset_data([], set([]))
        self.img3dGridModel.reset_data([], set([]))

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

    def cleanup_workers(self):
        """Stop all persistent workers before app exit."""
        self.showAlertSignal.emit("Important Alert", "Please wait as we safely close the app...")
        for worker in [self._gt_worker, self._ai_worker, self._hist_worker]:
            if worker:
                worker.stop()

    def _cancel_loading(self, worker_id):
        if worker_id == 1:
            self._stop_wait()

        if worker_id == 2:
            self._stop_ai_task()

        if worker_id == 3:
            self._stop_histogram_calculation()

    def _handle_progress_update(self, status_data: ProgressData) -> None:
        """
        Handler function for progress updates for ongoing GT tasks.
        Args:
            status_data: ProgressData object that contains the percentage and status message of the current task.

        Returns:

        """

        if status_data is None:
            return

        if 0 <= status_data.percent <= 100:
            if status_data.sender == "AI":
                self.updateAIProgressSignal.emit(status_data.percent, status_data.message)
            else:
                self.updateProgressSignal.emit(status_data.percent, status_data.message)
            logging.info(f"({status_data.sender}) {status_data.percent}%: {status_data.message}", extra={'user': 'SGT Logs'})

        if status_data.type == "info":
            if status_data.sender == "AI":
                self.updateAIProgressSignal.emit(101, status_data.message)
            else:
                self.updateProgressSignal.emit(101, status_data.message)
            logging.info(f"({status_data.sender}) {status_data.message}", extra={'user': 'SGT Logs'})
        elif status_data.type == "error":
            self.errorSignal.emit(status_data.message)
            logging.exception(f"({status_data.sender}) {status_data.message}", extra={'user': 'SGT Logs'})

    def _handle_finished(self, worker_id: int, success_val: bool, result: None | list | TaskResult) -> None:
        """
        Handler function for sending updates/signals on termination of tasks.
        Args:
            worker_id: The process worker ID.
            success_val: True if the task was successful, False otherwise.
            result: The result of the task.
        Returns:
            None
        """
        self._cancel_loading(worker_id)
        if not success_val:
            if type(result) is list:
                logging.info(result[0] + ": " + result[1], extra={'user': 'SGT Logs'})
                self.taskTerminatedSignal.emit(success_val, result)
        else:
            if isinstance(result, TaskResult):
                self.stop_current_task(worker_id, cancel_job=False)
                if result.task_id == "Export Graph" or result.task_id == "Save Images":
                    # Saving files to Output Folder
                    self._handle_progress_update(ProgressData(percent=100, sender="GT", message=f"Files Saved!"))
                    self.taskTerminatedSignal.emit(success_val, ["Files Saved", result.message])
                if result.task_id == "Rate Graph":
                    self._handle_progress_update(ProgressData(type="info", sender="AI", message=f"Graph image successfully uploaded!"))
                    self.taskTerminatedSignal.emit(success_val, ["Graph Rated", result.message])
                if result.task_id == "Extract Graph" or result.task_id == "Image Colors":
                    sgt_obj = self.get_selected_sgt_obj()
                    if result.task_id == "Image Colors":
                        sgt_obj.ntwk_p = result.data[0]
                        if result.data[1] is not None:
                            self.imgColorsModel.reset_data(result.data[1])
                    else:
                        sgt_obj.ntwk_p = result.data
                    self._handle_progress_update(ProgressData(percent=100, sender="GT", message=result.message))
                    # Update image configs
                    self.synchronize_img_models(sgt_obj)
                    # Update QML to visualize graph
                    self.changeImageSignal.emit()
                    # Update Graph & Compute properties
                    self.synchronize_graph_models(sgt_obj)
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(success_val, [])
                if result.task_id == "Compute GT":
                    self._handle_progress_update(ProgressData(percent=100, sender="GT", message=f"GT PDF successfully generated! Check it out in 'Output Dir'."))
                    self.update_sgt_obj(result.data)
                    sgt_obj = self.get_selected_sgt_obj()
                    # Update image configs
                    self.synchronize_img_models(sgt_obj)
                    # Update Graph & Compute properties
                    self.synchronize_graph_models(sgt_obj)
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(True,
                                                   ["GT calculations completed", "The image's GT parameters have been "
                                                                                 "calculated. Check out generated PDF in "
                                                                                 "'Output Dir'."])
                if result.task_id == "Compute Multi GT":
                    self._handle_progress_update(ProgressData(percent=100, sender="GT", message=f"All GT PDF successfully generated! Check it out in 'Output Dir'."))
                    self.update_sgt_obj(result.data)
                    sgt_obj = self.get_selected_sgt_obj()
                    # Update image configs
                    self.synchronize_img_models(sgt_obj)
                    # Update Graph & Compute properties
                    self.synchronize_graph_models(sgt_obj)
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(True, ["All GT calculations completed", "GT parameters of all "
                                                                                           "images have been calculated. Check "
                                                                                           "out all the generated PDFs in "
                                                                                           "'Output Dir'."])
                if result.task_id == "Metaheuristic Search":
                    # AI Mode search results (image configs)
                    if result.status == "Finished":
                        self._handle_progress_update(ProgressData(percent=100, sender="AI", message=f"Search completed!"))
                        sgt_obj = self.get_selected_sgt_obj()
                        sgt_obj.ntwk_p = result.data
                        # Update image configs and load Binary Image
                        self.synchronize_img_models(sgt_obj)
                        # Update Graph & Compute properties
                        self.synchronize_graph_models(sgt_obj)
                        self.apply_changes(view="binary")
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(success_val, [])
            elif type(result) is list:
                # Image histogram calculated
                self.stop_current_task(worker_id, cancel_job=False)
                if len(self._sgt_objs) > 0:
                    sgt_obj = self.get_selected_sgt_obj()
                    sel_img_batch = sgt_obj.ntwk_p.selected_batch
                    self.imgHistogramModel.reset_data(result, sel_img_batch.selected_images_idx)
                    self.imageChangedSignal.emit()  # trigger QML UI update
            else:
                self.taskTerminatedSignal.emit(success_val, [])

            # Auto-save changes to the project data file
            if len(self._sgt_objs.items()) <= 10:
                self.save_project_data()

    def _submit_job(self, worker_id, task_fxn, fxn_args=(), track_updates: bool = True) -> None:
        """Start a background thread and its associated worker."""

        def _sync_signals(bg_worker: PersistentProcessWorker):
            bg_worker.taskCompleted.connect(self._handle_finished)
            if track_updates:
                bg_worker.inProgress.connect(self._handle_progress_update)

        if task_fxn is None or worker_id is None:
            return

        base_funcs = BaseWorker()
        if task_fxn == "Calculate-Histogram":
            target = base_funcs.task_calculate_img_histogram
        elif task_fxn == "Retrieve-Colors":
            target = base_funcs.task_retrieve_img_colors
        elif task_fxn == "Eliminate-Colors":
            target = base_funcs.task_eliminate_img_colors
        elif task_fxn == "Extract-Graph":
            target = base_funcs.task_extract_graph
        elif task_fxn == "Compute-GT":
            target = base_funcs.task_compute_gt
        elif task_fxn == "Compute-Multi-GT":
            target = base_funcs.task_compute_multi_gt
        elif task_fxn == "Export-Graph":
            target = base_funcs.task_export_graph
        elif task_fxn == "Save-Images":
            target = base_funcs.task_save_images
        elif task_fxn == "Metaheuristic-Search":
            target = base_funcs.task_metaheuristic_search
        elif task_fxn == "Rate-Graph":
            target = base_funcs.task_rate_graph
        else:
            return

        if worker_id == 1:
            # base_funcs.attach_progress_queue(self._gt_worker.status_queue)
            started = self._gt_worker.submit_task(func=target, args=fxn_args)
            if not started:
                self.showAlertSignal.emit("Please Wait", "Another GT job is running!")
                return
            _sync_signals(self._gt_worker)
        elif worker_id == 2:
            started = self._ai_worker.submit_task(func=target, args=fxn_args)
            if not started:
                self.showAlertSignal.emit("Please Wait", "Another AI search is running!")
                return
            _sync_signals(self._ai_worker)
        elif worker_id == 3:
            started = self._hist_worker.submit_task(func=target, args=fxn_args)
            if not started:
                return
            _sync_signals(self._hist_worker)
        else:
            return

    @Slot(int)
    def stop_current_task(self, worker_id: int = 1, cancel_job: bool = True):
        """Stop a background thread and its associated worker."""
        # self.showAlertSignal.emit("Important Alert", "Cancelling job, please wait...")
        if worker_id == 1:
            if cancel_job:
                self._handle_progress_update(ProgressData(percent=99, sender="GT", message="Cancelling job, please wait..."))
            else:
                # Restart Process after 3 tasks
                if self._gt_worker.task_count < 3:
                    return
            # self._gt_worker.restart()
            self._gt_worker.stop()
            self._gt_worker = PersistentProcessWorker(worker_id)
            self._handle_finished(worker_id, True, None)

        if worker_id == 2:
            if cancel_job:
                self._handle_progress_update(ProgressData(percent=99, sender="AI", message="Cancelling job, please wait..."))
            else:
                if self._ai_worker.task_count < 3:
                    return
            # self._ai_worker.restart()
            self._ai_worker.stop()
            self._ai_worker = PersistentProcessWorker(worker_id)
            self._handle_finished(worker_id, True, None)

        if worker_id == 3:
            if cancel_job:
                self._handle_progress_update(ProgressData(percent=99, sender="GT", message="Cancelling job, please wait..."))
            else:
                if self._hist_worker.task_count < 3:
                    return
            # self._hist_worker.restart()
            self._hist_worker.stop()
            self._hist_worker = PersistentProcessWorker(worker_id)

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
            "A software tool for performing Graph/Network Theory analysis on <br> "
            "microscopy images. This is a modified version of StructuralGT <br> "
            "initially proposed by D. Vecchio <br> "
            "DOI: <a href='https://pubs.acs.org/doi/10.1021/acsnano.1c04711'>10.1021/acsnano.1c04711</a>."
            "<br></p><p>"
            "<b>Main Contributors:</b>"
            "<table border='0.5' cellspacing='0' cellpadding='4'>"
            # "<tr><th>Name</th><th>Email</th></tr>"
            "<tr><td>Dickson Owuor</td><td>owuor@umich.edu</td></tr>"
            "<tr><td>Nicolas Kotov</td><td>kotov@umich.edu</td></tr>"
            "<tr><td>Alain Kadar</td><td>alaink@umich.edu</td></tr>"
            "<tr><td>Xiong Ye Xiao</td><td>xiongyex@usc.edu</td></tr>"
            "<tr><td>Kotov Lab</td><td></td></tr>"
            "<tr><td>COMPASS</td><td></td></tr>"
            "</table>"
            "<br></p><p>"
            "<b>Documentation:</b> <a href='https://structural-gt.readthedocs.io'>structural-gt.readthedocs.io</a>"
            "</p><p>"
            f"<b> Version: </b> {self.get_sgt_version()}"
            "</p><p>"
            "<b>License:</b> GPL GNU v3"
            "</p><p>"
            "<b>Icon Acknowledgements:</b>"
            "<ol>"
            "<li> <a href='https://www.iconfinder.com/'>IconFinder Library</a></li>"
            "<li> <a href='https://www.flaticon.com/'>Flaticon</a> </li>"
            "</ol>"
            "</p><p><br>"
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

    @Slot()
    def reset_colors_model(self):
        """Erase existing data in the colors model."""
        self.reset_qml_models(only_colors=True)

    @Slot(int)
    def select_img_batch(self, batch_index=-1):
        if batch_index < 0:
            return

        try:
            sgt_obj = self.get_selected_sgt_obj()
            if sgt_obj is None:
                return
            sgt_obj.ntwk_p.select_image_batch(batch_index)

            # Load the SGT Object data of the selected image
            self.synchronize_img_models(sgt_obj)
            self.synchronize_graph_models(sgt_obj)
            self.reset_qml_models()

            # Trigger QML image update
            self.changeImageSignal.emit()
        except Exception as err:
            logging.exception("Batch Change Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Image Batch Error", f"Error encountered while trying to access batch "
                                                           f"{batch_index}. Restart app and try again.")

    @Slot(int, str, result=str)
    def get_selected_image(self, img_pos: int = 0, view: str = "original") -> str:
        b64_img = super().get_selected_image(img_pos, view)
        return b64_img

    @Slot(int, bool)
    def toggle_selected_batch_image(self, img_index, selected):
        sgt_obj = self.get_selected_sgt_obj()
        sel_img_batch = sgt_obj.ntwk_p.selected_batch
        if selected:
            sel_img_batch.selected_images_idx.add(img_index)
        else:
            sel_img_batch.selected_images_idx.discard(img_index)
        self.changeImageSignal.emit()

    @Slot(int)
    def select_batch_image_index(self, img_pos=-1):
        if img_pos < 0:
            return

        try:
            sgt_obj = self.get_selected_sgt_obj()
            if sgt_obj is None:
                return

            sel_img_batch = sgt_obj.ntwk_p.selected_batch
            for img_idx in sel_img_batch.selected_images_idx:
                sel_img_batch.selected_images_idx.discard(img_idx)
            sel_img_batch.selected_images_idx.add(img_pos)

            # Trigger QML image update
            self.reset_qml_models()
            self.imageChangedSignal.emit()
        except Exception as err:
            logging.exception(f"Image Index Error: {err}", extra={'user': 'SGT Logs'})

    @Slot(result=int)
    def get_selected_batch_image_index(self):
        sel_pos = 0
        try:
            sgt_obj = self.get_selected_sgt_obj()
            if sgt_obj is None:
                return sel_pos

            sel_img_batch = sgt_obj.ntwk_p.selected_batch
            for i, img_idx in enumerate(sel_img_batch.selected_images_idx):
                if i == 0:
                    sel_pos = img_idx
                else:
                    sel_img_batch.selected_images_idx.discard(img_idx)
        except Exception as err:
            logging.exception("Image Index Error: %s", err, extra={'user': 'SGT Logs'})
            sel_pos = 0
        return sel_pos

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
            sgt_obj = self.get_selected_sgt_obj()
            self.reset_qml_models()
            self.synchronize_img_models(sgt_obj)
            self.synchronize_graph_models(sgt_obj)
            self.imgThumbnailModel.set_selected(self._selected_sgt_obj_index)
            # Load the selected image into the view
            self.changeImageSignal.emit()
            # Run AI search (if enabled)
            self.run_ai_filter_search()
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
    def apply_changes(self, view: str = ""):
        """Retrieve changes made by the user and apply to image/graph."""
        if not self._applying_changes:  # Disallow concurrent changes
            self._applying_changes = True
            if view != "":
                sgt_obj = self.get_selected_sgt_obj()
                sgt_obj.ntwk_p.selected_batch_view = view
            self.changeImageSignal.emit()

    @Slot(bool, str, int)
    def undo_applied_changes(self, undo: bool = True, change_type: str = "cropping", img_idx: int = -1):
        if undo:
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.undo_img_changes(img_pos=img_idx)

            # Emit signal to update UI with new image
            self.changeImageSignal.emit()
            if change_type == "cropping":
                self.showUnCroppingToolSignal.emit(False)

    @Slot()
    def compute_img_histogram(self):
        """Calculate the histogram of the image."""
        if self._wait_flag_hist:
            return

        try:
            self._start_histogram_calculation()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(3, "Calculate-Histogram", (sgt_obj.ntwk_p,), False)
        except Exception as err:
            self._stop_histogram_calculation()
            logging.exception("Histogram Calculation Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_finished(3, False, ["Histogram Calculation Failed", "Unable to calculate image histogram!"])

    @Slot()
    def apply_img_scaling(self):
        """Retrieve settings from the model and send to Python."""
        try:
            # Apply scaling
            self.set_auto_scale(True)
            sgt_obj = self.get_selected_sgt_obj()
            sgt_obj.ntwk_p.auto_scale = self._allow_auto_scale
            sgt_obj.ntwk_p.apply_img_scaling()

            # Update properties and load the scaled image into view
            self.imagePropsModel.reset_data(sgt_obj.ntwk_p.selected_batch.props)
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
        self._handle_progress_update(ProgressData(percent=0, sender="GT", message=f"Exporting Graph Data..."))
        try:
            if self.get_selected_sgt_obj().ntwk_p.selected_batch.is_graph_only:
                return

            self._handle_progress_update(ProgressData(percent=20, sender="GT", message=f"Exporting Graph Data..."))
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(1, "Export-Graph", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            logging.exception("Unable to Export Graph: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False,
                                           ["Unable to Export Graph", "Error exporting graph to file. Try again."])

    @Slot()
    def save_img_files(self):
        """Retrieve and save images to the file."""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self._handle_progress_update(ProgressData(percent=0, sender="GT", message=f"Saving images..."))
        try:
            if self.get_selected_sgt_obj().ntwk_p.selected_batch.is_graph_only:
                return

            self._handle_progress_update(ProgressData(percent=10, sender="GT", message=f"Saving images..."))
            sel_images = self.get_selected_images()
            for val in self.saveImgModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]

            self._handle_progress_update(ProgressData(percent=20, sender="GT", message=f"Saving images..."))
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(1, "Save-Images", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            logging.exception("Unable to Save Image Files: " + str(err), extra={'user': 'SGT Logs'})
            self.taskTerminatedSignal.emit(False,
                                           ["Unable to Save Image Files", "Error saving images to file. Try again."])

    @Slot(int, int)
    def run_retrieve_img_colors(self, img_pos: int, max_colors: int):
        """Retrieve the dominant colors of the image."""
        if self._wait_flag:
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self._start_wait(msg="image_filters")
            ntwk_p = self.get_selected_sgt_obj().ntwk_p
            self._submit_job(1, "Retrieve-Colors", (ntwk_p, img_pos, max_colors), True)
        except Exception as err:
            self._stop_wait()
            logging.exception(f"Retrieve Colors Error: {err}", extra={'user': 'SGT Logs'})
            self._handle_progress_update(ProgressData(type="error", sender="GT", message=f"Unable to retrieve colors! Try again."))
            self._handle_finished(1, False, ["Get Colors Failed", "Unable to retrieve dominant colors!"])

    @Slot(int, int)
    def run_eliminate_img_colors(self, img_pos: int, swap_white: int):
        """Eliminate selected image colors by swapping the values of pixels where they appear."""
        if self._wait_flag:
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self._start_wait(msg="image_filters")
            ntwk_p = self.get_selected_sgt_obj().ntwk_p
            colors = ntwk_p.image_obj.dominant_colors

            # Update ImageProcessor object
            for val in self.imgColorsModel.list_data:
                for color in colors:
                    if color.hex_code == val["text"]:
                        color.is_selected = True if val["value"] == 1 else False

            self._submit_job(1, "Eliminate-Colors", (ntwk_p, img_pos, swap_white), True)
        except Exception as err:
            self._stop_wait()
            logging.exception(f"Eliminate Colors Error: {err}", extra={'user': 'SGT Logs'})
            self._handle_progress_update(ProgressData(type="error", sender="GT", message=f"Unable to eliminate colors! Try again."))
            self._handle_finished(1, False, ["Eliminate Colors Failed", "Unable to eliminate colors!"])

    @Slot()
    def run_extract_graph(self):
        """Retrieve settings from the model and send to Python."""

        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(1, "Extract-Graph", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            self._stop_wait()
            logging.exception("Graph Extraction Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_progress_update(ProgressData(type="error", sender="GT", message=f"Fatal error occurred! Close the app and try again."))
            self._handle_finished(1,False, ["Graph Extraction Error",
                                          "Fatal error while trying to extract graph. "
                                          "Close the app and try again."])

    @Slot(float)
    def rate_graph(self, rating: float):
        """Rate extracted graph on a scale of 1-10"""
        if self._wait_flag_ai:
            logging.info("Please wait for AI task to finish.", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "AI task is still running!")
            return

        try:
            self._start_ai_task()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(2, "Rate-Graph", (rating, sgt_obj.ntwk_p,), True)
        except Exception as err:
            self._stop_ai_task()
            logging.info("Rate Graph Error: " + str(err), extra={'user': 'SGT Logs'})

    @Slot()
    def run_graph_analyzer(self):
        """Retrieve settings from the model and send to Python."""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self._start_wait()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(1, "Compute-GT", (sgt_obj,), True)
        except Exception as err:
            self._stop_wait()
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_progress_update(ProgressData(type="error", sender="GT", message=f"Fatal error occurred! Close the app and try again."))
            self._handle_finished(1, False, ["GT Computation Error",
                                          "Fatal error while trying calculate GT parameters. "
                                          "Close the app and try again."])

    @Slot()
    def run_multi_graph_analyzer(self):
        """"""
        if self._wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self._start_wait()
            # Update Configs
            self.replicate_sgt_configs()
            # Start Background Process
            self._submit_job(1, "Compute-Multi-GT", (self._sgt_objs,), True)
        except Exception as err:
            self._stop_wait()
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self._handle_progress_update(ProgressData(type="error", sender="GT", message=f"Fatal error occurred! Close the app and try again."))
            self._handle_finished(1,False, ["GT Computation Error",
                                          "Fatal error while trying calculate GT parameters. "
                                          "Close the app and try again."])

    @Slot()
    def run_ai_filter_search(self):
        """Run AI filter search on the selected SGT object."""
        if not self._ai_mode_active:
            return

        if self._wait_flag_ai:
            logging.info("Another AI task is running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another AI task is running!")
            return

        try:
            self._start_ai_task()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(2, "Metaheuristic-Search", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            self._stop_ai_task()
            logging.info("AI Mode Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot()
    def reset_ai_filter_results(self):
        """Reset the results by moving the best candidate to the ignore list"""
        sgt_obj = self.get_selected_sgt_obj()
        sgt_obj.ntwk_p.reset_metaheuristic_search()
        self.run_ai_filter_search()

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

    @Slot(int, int, int, int, int, int)
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
            self.showAlertSignal.emit("Cropping Error",
                                      "Error occurred while cropping image. Close the app and try again.")

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
            sgt_obj = self.get_selected_sgt_obj()
            self.synchronize_img_models(sgt_obj)
            self.synchronize_graph_models(sgt_obj)
            self.load_image(reload_thumbnails=True)
        return is_successful

    @Slot(str, result=bool)
    def upload_multiple_images(self, img_dir_path):
        """
        Verify and validate multiple image paths, use each to create an SGT object, then load the last one in view.
        """
        is_successful = self.add_multiple_images(img_dir_path)
        if is_successful:
            sgt_obj = self.get_selected_sgt_obj()
            self.synchronize_img_models(sgt_obj)
            self.synchronize_graph_models(sgt_obj)
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
            # self._project_open = False
            logging.exception("Create Project Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Create Project Error",
                                      "Failed to create SGT project. Close the app and try again.")
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
