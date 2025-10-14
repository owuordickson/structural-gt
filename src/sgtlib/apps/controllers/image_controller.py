# SPDX-License-Identifier: GNU GPL v3
"""
Pyside6 (GUI components) controller class for applying image filters.
"""

import logging
import numpy as np
from PySide6.QtCore import Signal, Slot, QObject, Property

from ..models.table_model import TableModel
from ..models.checkbox_model import CheckBoxModel
from ..models.imagegrid_model import ImageGridModel
from ...utils.sgt_utils import ProgressData
from ...compute.graph_analyzer import GraphAnalyzer


class ImageController(QObject):

    showImageFilterControls = Signal(bool)
    showCroppingControls = Signal(bool)

    _imgFiltersBusyChanged = Signal()
    _histogramBusyChanged = Signal()

    def __init__(self, controller_obj, parent: QObject = None):
        super().__init__(parent)
        self._ctrl = controller_obj
        self._img_loaded = False
        self._applying_changes = False
        self._allow_auto_scale = True
        self._wait_flag_hist = False
        self._wait_flag_filters = False

        # Create Models
        self.imagePropsModel = TableModel([])
        self.imgBatchModel = CheckBoxModel([])
        self.imgControlModel = CheckBoxModel([])
        self.imgBinFilterModel = CheckBoxModel([])
        self.microscopyPropsModel = CheckBoxModel([])
        self.imgFilterModel = CheckBoxModel([])
        self.imgColorsModel = CheckBoxModel([])

        self.imgScaleOptionModel = CheckBoxModel([])
        self.imgViewOptionModel = CheckBoxModel([])
        self.saveImgModel = CheckBoxModel([])
        self.img3dGridModel = ImageGridModel([], set([]))
        self.imgHistogramModel = ImageGridModel([], set([]))

        # Attach listener for syncing models
        self._ctrl.syncModelSignal.connect(self.synchronize_img_models)

    @Property(bool, notify=_imgFiltersBusyChanged)
    def img_filters_busy(self):
        return self._wait_flag_filters

    @Property(bool, notify=_histogramBusyChanged)
    def histogram_busy(self):
        return self._wait_flag_hist

    def start_task(self):
        """Activate the wait flag and send a wait signal."""
        self._ctrl.wait_flag = True
        self._wait_flag_filters = True
        self._imgFiltersBusyChanged.emit()

    def stop_task(self):
        """Deactivate the wait flag and send a wait signal."""
        self._ctrl.wait_flag = False
        self._wait_flag_filters = False
        self._imgFiltersBusyChanged.emit()

    def start_histogram_calculation(self):
        """Start computing the histogram of the selected image."""
        self._wait_flag_hist = True
        self._histogramBusyChanged.emit()

    def stop_histogram_calculation(self):
        """Stop computing the histogram of the selected image."""
        self._wait_flag_hist = False
        self._histogramBusyChanged.emit()

    def synchronize_img_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload image configuration selections and controls from saved dict to QML gui_mcw after the image is loaded.

            :param sgt_obj: A GraphAnalyzer object with all saved user-selected configurations.
        """
        if sgt_obj is None:
            self._ctrl.is_syncing = False
            return

        try:
            # Models Auto-update with saved sgt_obj configs. No need to re-assign!
            ntwk_p = sgt_obj.ntwk_p
            sel_img_batch = ntwk_p.selected_batch
            options_img = ntwk_p.image_obj.configs

            # Get data from object configs
            img_controls = [v for v in options_img.values() if v["type"] == "image-control"]
            bin_filters = [v for v in options_img.values() if v["type"] == "binary-filter"]
            img_filters = [v for v in options_img.values() if v["type"] == "image-filter"]
            img_properties = [v for v in options_img.values() if v["type"] == "image-property"]
            file_options = [v for v in options_img.values() if v["type"] == "file-options"]

            batch_list = [{"id": f"batch_{i}", "text": f" Batch {i + 1}", "value": i}
                          for i in range(len(sgt_obj.ntwk_p.image_batches))]

            # Update QML adapter-models with fetched data
            self.imgBatchModel.reset_data(batch_list)
            self.imgScaleOptionModel.reset_data(sel_img_batch.scaling_options)
            self.imgViewOptionModel.reset_data(sel_img_batch.view_options)

            self.imgControlModel.reset_data(img_controls)
            self.imgBinFilterModel.reset_data(bin_filters)
            self.imgFilterModel.reset_data(img_filters)
            self.saveImgModel.reset_data(file_options)

            self.microscopyPropsModel.reset_data(img_properties)
            self.imagePropsModel.reset_data(sel_img_batch.props)
            self._ctrl.is_syncing = False
        except Exception as err:
            self._ctrl.is_syncing = False
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Fatal Error", "Error re-loading image configurations! Close app and try again.")

    def reset_img_models(self, only_colors: bool = False):
        """
        Reset some of the CheckBox models when different image is loaded.

        Args:
            only_colors: If True, reset only the imgColorsModel model. If False, reset all models.

        Returns:
            None
        """
        # Erase existing data in QML adapter-models
        self.imgColorsModel.reset_data([])
        if only_colors:
            return
        self.imgHistogramModel.reset_data([], set([]))

    @Slot(result=bool)
    def display_image(self):
        return self._img_loaded

    @Slot(result=bool)
    def is_img_3d(self):
        sgt_obj = self._ctrl.get_selected_sgt_obj()
        if sgt_obj is None:
            return False
        sel_img_batch = sgt_obj.ntwk_p.selected_batch
        is_3d = not sel_img_batch.is_2d
        return is_3d

    @Slot(result=bool)
    def enable_img_controls(self):
        """Enable image controls."""
        if len(self._ctrl.sgt_objs) <= 0:
            return False

        sgt_obj = self._ctrl.get_selected_sgt_obj()
        if sgt_obj is None:
            return False
        return not sgt_obj.ntwk_p.selected_batch.is_graph_only

    @Slot(result=bool)
    def image_batches_exist(self):
        if not self._img_loaded:
            return False

        sgt_obj = self._ctrl.get_selected_sgt_obj()
        batch_count = len(sgt_obj.ntwk_p.image_batches)
        batches_exist = True if batch_count > 1 else False
        return batches_exist

    @Slot(result=str)
    def get_pixmap(self):
        """Returns the URL that QML should use to load the image"""
        curr_img_view = np.random.randint(0, 4)
        unique_num = self._ctrl.selected_sgt_obj_index + curr_img_view + np.random.randint(low=21, high=1000)
        return "image://imageProvider/" + str(unique_num)

    @Slot(result=int)
    def get_selected_img_batch(self):
        try:
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            return sgt_obj.ntwk_p.selected_batch_index
        except AttributeError:
            logging.exception("No image added! Please add at least one image.", extra={'user': 'SGT Logs'})
            return 0

    @Slot(result=bool)
    def get_auto_scale(self):
        return self._allow_auto_scale

    @Slot(int, str, result=str)
    def get_selected_image(self, img_pos: int = 0, view: str = "original") -> str:
        b64_img = self._ctrl.get_selected_image(img_pos, view)
        return b64_img

    @Slot(result=int)
    def get_selected_batch_image_index(self):
        sel_pos = 0
        try:
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            if sgt_obj is None:
                return sel_pos

            sel_pos = sgt_obj.ntwk_p.selected_batch.selected_frame_pos
        except Exception as err:
            logging.exception("Image Index Error: %s", err, extra={'user': 'SGT Logs'})
            sel_pos = 0
        return sel_pos

    @Slot(bool)
    def set_auto_scale(self, auto_scale):
        """Set the auto-scale parameter for each image."""
        self._allow_auto_scale = auto_scale

    @Slot(int)
    def select_img_batch(self, batch_index=-1):
        if batch_index < 0:
            return

        try:
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            if sgt_obj is None:
                return
            sgt_obj.ntwk_p.select_image_batch(batch_index)

            # Trigger sync models and image refresh
            self._ctrl.is_syncing = True
            self._ctrl.syncModelSignal.emit(sgt_obj)
            self.reset_img_models()
            # while self._ctrl.is_syncing:
            #    print("waiting for sync to finish...")
            #    logging.info("Waiting for sync to finish...", extra={'user': 'SGT Logs'})
            self._ctrl.changeImageSignal.emit()
        except Exception as err:
            logging.exception("Batch Change Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Image Batch Error", f"Error encountered while trying to access batch "
                                                           f"{batch_index}. Restart app and try again.")

    @Slot(int)
    def select_batch_image_index(self, img_pos=-1):
        if img_pos < 0:
            return

        try:
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            if sgt_obj is None:
                return
            sgt_obj.ntwk_p.selected_batch.selected_frame_pos = img_pos

            # Trigger QML image update
            self.reset_img_models()
            self._ctrl.imageChangedSignal.emit()
        except Exception as err:
            logging.exception(f"Image Index Error: {err}", extra={'user': 'SGT Logs'})

    @Slot(int, bool)
    def toggle_selected_batch_image(self, img_index, selected):
        sgt_obj = self._ctrl.get_selected_sgt_obj()
        sel_img_batch = sgt_obj.ntwk_p.selected_batch
        if selected:
            sel_img_batch.selected_images_positions.add(img_index)
        else:
            sel_img_batch.selected_images_positions.discard(img_index)
        self._ctrl.changeImageSignal.emit()

    @Slot(str)
    def apply_changes(self, view: str = ""):
        """Retrieve changes made by the user and apply to image/graph."""
        if not self._applying_changes:  # Disallow concurrent changes
            self._applying_changes = True
            if view != "":
                sgt_obj = self._ctrl.get_selected_sgt_obj()
                sgt_obj.ntwk_p.selected_batch_view = view
            self._ctrl.changeImageSignal.emit()

    @Slot(bool, str, int)
    def undo_applied_changes(self, undo: bool = True, change_type: str = "cropping", img_idx: int = -1):
        if undo:
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            sgt_obj.ntwk_p.undo_img_changes(img_pos=img_idx)

            # Emit signal to update UI with new image
            self._ctrl.changeImageSignal.emit()
            if change_type == "cropping":
                self.showCroppingControls.emit(True)

    @Slot(int)
    def compute_img_histogram(self, img_pos: int):
        """Calculate the histogram of the image."""
        if self._wait_flag_hist:
            return

        try:
            self.start_histogram_calculation()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(3, "Calculate-Histogram", (sgt_obj.ntwk_p, img_pos), False)
        except Exception as err:
            self.stop_histogram_calculation()
            logging.exception("Histogram Calculation Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.handle_finished(3, False, ["Histogram Calculation Failed", "Unable to calculate image histogram!"])

    @Slot()
    def apply_img_scaling(self):
        """Retrieve settings from the model and send to Python."""
        try:
            # Apply scaling
            self.set_auto_scale(True)
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            sgt_obj.ntwk_p.auto_scale = self._allow_auto_scale
            sgt_obj.ntwk_p.apply_img_scaling()

            # Update properties and load the scaled image into view
            self.imagePropsModel.reset_data(sgt_obj.ntwk_p.selected_batch.props)
            self._ctrl.changeImageSignal.emit()
        except Exception as err:
            logging.exception("Apply Image Scaling: " + str(err), extra={'user': 'SGT Logs'})
            self._ctrl.handle_finished(-1, False, ["Unable to Rescale Image", "Error while tying to re-scale "
                                                                              "image. Try again."])

    @Slot(int, int, int, int, int, int)
    def crop_image(self, x, y, crop_width, crop_height, qimg_width, qimg_height):
        """Crop image using PIL and save it."""
        try:
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            sgt_obj.ntwk_p.crop_image(x, y, crop_width, crop_height, qimg_width, qimg_height)

            # Emit signal to update UI with new image
            self._ctrl.changeImageSignal.emit()
            self.showCroppingControls.emit(False)
        except Exception as err:
            logging.exception("Cropping Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Cropping Error",
                                      "Error occurred while cropping image. Close the app and try again.")

    @Slot(int)
    def save_cropped_image(self, img_pos: int):
        """Save cropped image to file."""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(1, "Save-Cropped-Image", (sgt_obj.ntwk_p, img_pos), False)
        except Exception as err:
            self.stop_task()
            logging.exception("Unable to Save Cropped Image: " + str(err), extra={'user': 'SGT Logs'})
            self._ctrl.handle_finished(-1, False, ["Unable to Save Cropped Image", "Error saving cropped image to file. Try again."])

    @Slot()
    def save_img_files(self):
        """Retrieve and save images to the file."""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another Task Running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        self._ctrl.handle_progress_update(ProgressData(percent=0, sender="GT", message=f"Saving images..."))
        try:
            if self._ctrl.get_selected_sgt_obj().ntwk_p.selected_batch.is_graph_only:
                return

            self._ctrl.handle_progress_update(ProgressData(percent=10, sender="GT", message=f"Saving images..."))
            sel_images = self._ctrl.get_selected_images()
            for val in self.saveImgModel.list_data:
                for img in sel_images:
                    img.configs[val["id"]]["value"] = val["value"]

            self._ctrl.handle_progress_update(ProgressData(percent=20, sender="GT", message=f"Saving images..."))
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(1, "Save-Images", (sgt_obj.ntwk_p, None), True)
        except Exception as err:
            self.stop_task()
            logging.exception("Unable to Save Image Files: " + str(err), extra={'user': 'SGT Logs'})
            self._ctrl.handle_finished(-1, False,
                                           ["Unable to Save Image Files", "Error saving images to file. Try again."])

    @Slot(int, int)
    def run_retrieve_img_colors(self, img_pos: int, max_colors: int):
        """Retrieve the dominant colors of the image."""
        if self._ctrl.wait_flag:
            self._ctrl.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self.start_task()
            ntwk_p = self._ctrl.get_selected_sgt_obj().ntwk_p
            self._ctrl.submit_job(1, "Retrieve-Colors", (ntwk_p, img_pos, max_colors), True)
        except Exception as err:
            self.stop_task()
            logging.exception(f"Retrieve Colors Error: {err}", extra={'user': 'SGT Logs'})
            self._ctrl.handle_progress_update(
                ProgressData(type="error", sender="GT", message=f"Unable to retrieve colors! Try again."))
            self._ctrl.handle_finished(1, False, ["Get Colors Failed", "Unable to retrieve dominant colors!"])

    @Slot(int, int)
    def run_eliminate_img_colors(self, img_pos: int, swap_white: int):
        """Eliminate selected image colors by swapping the values of pixels where they appear."""
        if self._ctrl.wait_flag:
            self._ctrl.showAlertSignal.emit("Please Wait", "Another Task Running!")
            return

        try:
            self.start_task()
            ntwk_p = self._ctrl.get_selected_sgt_obj().ntwk_p
            colors = ntwk_p.image_obj.dominant_colors

            # Update ImageProcessor object
            for val in self.imgColorsModel.list_data:
                for color in colors:
                    if color.hex_code == val["text"]:
                        color.is_selected = True if val["value"] == 1 else False

            self._ctrl.submit_job(1, "Eliminate-Colors", (ntwk_p, img_pos, swap_white), True)
        except Exception as err:
            self.stop_task()
            logging.exception(f"Eliminate Colors Error: {err}", extra={'user': 'SGT Logs'})
            self._ctrl.handle_progress_update(
                ProgressData(type="error", sender="GT", message=f"Unable to eliminate colors! Try again."))
            self._ctrl.handle_finished(1, False, ["Eliminate Colors Failed", "Unable to eliminate colors!"])
