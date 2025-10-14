# SPDX-License-Identifier: GNU GPL v3
"""
Pyside6 (GUI components) main controller class.
"""

import logging
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Slot, Signal, Property

from .ai_controller import AIController
from .base_controller import BaseController
from .graph_controller import GraphController
from .image_controller import ImageController
from .project_controller import ProjectController

from ..workers.persistent_worker import PersistentProcessWorker
from ..workers.base_workers import BaseWorker
from ...utils.sgt_utils import TaskResult, ProgressData


class MainController(BaseController):
    """Exposes a method to refresh the image in QML"""

    _waitChanged = Signal()
    _waitTextChanged = Signal()
    errorSignal = Signal(str)
    changeImageSignal = Signal()
    imageChangedSignal = Signal()
    syncModelSignal = Signal(object)
    updateProgressSignal = Signal(int, str)
    taskTerminatedSignal = Signal(bool, list)

    def __init__(self, qml_app: QApplication):
        super().__init__()
        self._qml_app = qml_app
        self.is_syncing = False

        # Add Controllers
        self.proj_ctrl = ProjectController(self)
        self.img_ctrl = ImageController(self)
        self.graph_ctrl = GraphController(self)
        self.ai_ctrl = AIController(self)

        # Create Persistent Workers (Processes)
        self._gt_worker = PersistentProcessWorker(worker_id=1)
        self._ai_worker = PersistentProcessWorker(worker_id=2)
        self._hist_worker = PersistentProcessWorker(worker_id=3)

    @property
    def qml_app(self):
        return self._qml_app

    @BaseController.wait_flag.setter
    def wait_flag(self, value: bool):
        """Sets the wait flag indicating if the application is currently running a task in the background."""
        self._wait_flag = value
        self._waitChanged.emit()

    @BaseController.wait_msg.setter
    def wait_msg(self, value: str):
        """Sets the wait message indicating the current task."""
        self._wait_msg = value
        self._waitTextChanged.emit()

    @Property(bool, notify=_waitChanged)
    def wait(self):
        return self._wait_flag

    @Property(str, notify=_waitTextChanged)
    def wait_text(self):
        return self._wait_msg

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
                img_list, img_cache = self.proj_ctrl.get_thumbnail_list()
                self.proj_ctrl.imgThumbnailModel.update_data(img_list, img_cache)

            # Load the SGT Object data of the selected image
            self.syncModelSignal.emit(self.get_selected_sgt_obj())
            self.proj_ctrl.imgThumbnailModel.set_selected(self._selected_sgt_obj_index)
            self.img_ctrl.reset_img_models()

            # Load the selected image into the view
            self.changeImageSignal.emit()

            # Run AI search (if enabled)
            self.ai_ctrl.run_ai_filter_search()
        except Exception as err:
            self.delete_sgt_object()
            self._selected_sgt_obj_index = 0
            logging.exception("Image Loading Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Image Error", "Error loading image. Try again.")

    def _cancel_loading(self, worker_id):
        if worker_id == 1:
            self.graph_ctrl.stop_task()
            self.img_ctrl.stop_task()
            self.graph_ctrl.stop_task()

        if worker_id == 2:
            self.ai_ctrl.stop_task()

        if worker_id == 3:
            self.img_ctrl.stop_histogram_calculation()

    def handle_progress_update(self, status_data: ProgressData) -> None:
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
                self.ai_ctrl.updateAIProgressSignal.emit(status_data.percent, status_data.message)
            else:
                self.updateProgressSignal.emit(status_data.percent, status_data.message)
            logging.info(f"({status_data.sender}) {status_data.percent}%: {status_data.message}", extra={'user': 'SGT Logs'})

        if status_data.type == "info":
            if status_data.sender == "AI":
                self.ai_ctrl.updateAIProgressSignal.emit(101, status_data.message)
            else:
                self.updateProgressSignal.emit(101, status_data.message)
            logging.info(f"({status_data.sender}) {status_data.message}", extra={'user': 'SGT Logs'})
        elif status_data.type == "error":
            self.errorSignal.emit(status_data.message)
            logging.exception(f"({status_data.sender}) {status_data.message}", extra={'user': 'SGT Logs'})

    def handle_finished(self, worker_id: int, success_val: bool, result: None | list | TaskResult) -> None:
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
                    self.handle_progress_update(ProgressData(percent=100, sender="GT", message=f"Files Saved!"))
                    self.taskTerminatedSignal.emit(success_val, ["Files Saved", result.message])
                if result.task_id == "Rate Graph":
                    self.handle_progress_update(ProgressData(type="info", sender="AI", message=f"Graph image successfully uploaded!"))
                    self.taskTerminatedSignal.emit(success_val, ["Graph Rated", result.message])
                if result.task_id == "Extract Graph" or result.task_id == "Image Colors":
                    sgt_obj = self.get_selected_sgt_obj()
                    if result.task_id == "Image Colors":
                        sgt_obj.ntwk_p = result.data[0]
                        if result.data[1] is not None:
                            self.img_ctrl.imgColorsModel.reset_data(result.data[1])
                    else:
                        sgt_obj.ntwk_p = result.data
                    self.handle_progress_update(ProgressData(percent=100, sender="GT", message=result.message))
                    # Sync models and refresh image
                    self.syncModelSignal.emit(sgt_obj)
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(success_val, [])
                if result.task_id == "Compute GT":
                    self.handle_progress_update(ProgressData(percent=100, sender="GT", message=f"GT PDF successfully generated! Check it out in 'Output Dir'."))
                    self.update_sgt_obj(result.data)
                    sgt_obj = self.get_selected_sgt_obj()
                    # Sync models and refresh image
                    self.syncModelSignal.emit(sgt_obj)
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(True,
                                                   ["GT calculations completed", "The image's GT parameters have been "
                                                                                 "calculated. Check out generated PDF in "
                                                                                 "'Output Dir'."])
                if result.task_id == "Compute Multi GT":
                    self.handle_progress_update(ProgressData(percent=100, sender="GT", message=f"All GT PDF successfully generated! Check it out in 'Output Dir'."))
                    self.update_sgt_obj(result.data)
                    sgt_obj = self.get_selected_sgt_obj()
                    # Sync models and refresh image
                    self.syncModelSignal.emit(sgt_obj)
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(True, ["All GT calculations completed", "GT parameters of all "
                                                                                           "images have been calculated. Check "
                                                                                           "out all the generated PDFs in "
                                                                                           "'Output Dir'."])
                if result.task_id == "Metaheuristic Search":
                    # AI Mode search results (image configs)
                    if result.status == "Finished":
                        self.handle_progress_update(ProgressData(percent=100, sender="AI", message=f"Search completed!"))
                        sgt_obj = self.get_selected_sgt_obj()
                        sgt_obj.ntwk_p = result.data
                        # Sync models and refresh image
                        self.syncModelSignal.emit(sgt_obj)
                        self.img_ctrl.apply_changes(view="binary")
                    # Send task termination signal to QML
                    self.taskTerminatedSignal.emit(success_val, [])
            elif type(result) is list:
                # Image histogram calculated
                self.stop_current_task(worker_id, cancel_job=False)
                if len(self._sgt_objs) > 0:
                    if result is not None:
                        self.img_ctrl.imgHistogramModel.reset_data(result, set([]))
                    self.imageChangedSignal.emit()  # trigger QML UI update
            else:
                self.taskTerminatedSignal.emit(success_val, [])

            # Auto-save changes to the project data file
            if len(self._sgt_objs.items()) <= 10:
                self.proj_ctrl.save_project_data()

    def submit_job(self, worker_id, task_fxn, fxn_args=(), track_updates: bool = True) -> None:
        """Start a background thread and its associated worker."""

        def _sync_signals(bg_worker: PersistentProcessWorker):
            bg_worker.taskCompleted.connect(self.handle_finished)
            if track_updates:
                bg_worker.inProgress.connect(self.handle_progress_update)

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
        elif task_fxn == "Save-Images" or task_fxn == "Save-Cropped-Image":
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

    def delete_sgt_object(self, index=None):
        """
        Delete SGT Obj stored at the specified index (if not specified, get the current index).
        """
        deleted = super().delete_sgt_object(index=index)
        if deleted:
            # Update Data
            img_list, img_cache = self.proj_ctrl.get_thumbnail_list()
            self.proj_ctrl.imgThumbnailModel.update_data(img_list, img_cache)
            self.img_ctrl.imagePropsModel.reset_data([])
            self.graph_ctrl.graphPropsModel.reset_data([])
            self.graph_ctrl.graphComputeModel.reset_data([])
            self._selected_sgt_obj_index = 0
            self.load_image(reload_thumbnails=True)
            self.imageChangedSignal.emit()

    def cleanup_workers(self):
        """Stop all persistent workers before app exit."""
        self.showAlertSignal.emit("Important Alert", "Please wait as we safely close the app...")
        for worker in [self._gt_worker, self._ai_worker, self._hist_worker]:
            if worker:
                worker.stop()

    @Slot(int)
    def stop_current_task(self, worker_id: int = 1, cancel_job: bool = True):
        """Stop a background thread and its associated worker."""
        # self.showAlertSignal.emit("Important Alert", "Cancelling job, please wait...")
        if worker_id == 1:
            if cancel_job:
                self.handle_progress_update(ProgressData(percent=99, sender="GT", message="Cancelling job, please wait..."))
            else:
                # Restart Process after 3 tasks
                if self._gt_worker.task_count < 3:
                    return
            # self._gt_worker.restart()
            self._gt_worker.stop()
            self._gt_worker = PersistentProcessWorker(worker_id)
            self.handle_finished(worker_id, True, None)

        if worker_id == 2:
            if cancel_job:
                self.handle_progress_update(ProgressData(percent=99, sender="AI", message="Cancelling job, please wait..."))
            else:
                if self._ai_worker.task_count < 3:
                    return
            # self._ai_worker.restart()
            self._ai_worker.stop()
            self._ai_worker = PersistentProcessWorker(worker_id)
            self.handle_finished(worker_id, True, None)

        if worker_id == 3:
            if cancel_job:
                self.handle_progress_update(ProgressData(percent=99, sender="GT", message="Cancelling job, please wait..."))
            else:
                if self._hist_worker.task_count < 3:
                    return
            # self._hist_worker.restart()
            self._hist_worker.stop()
            self._hist_worker = PersistentProcessWorker(worker_id)

    @Slot(result=bool)
    def is_task_running(self):
        return self._wait_flag
