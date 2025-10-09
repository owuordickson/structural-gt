# SPDX-License-Identifier: GNU GPL v3

"""
Base controller class for StructuralGT.
"""

import os
import logging
from PySide6.QtCore import Signal, QObject

from ...utils.sgt_utils import verify_path, img_to_base64
from ...imaging.image_processor import ImageProcessor, ALLOWED_IMG_EXTENSIONS
from ...compute.graph_analyzer import GraphAnalyzer

class BaseController(QObject):

    showAlertSignal = Signal(str, str)

    def __init__(self, config_file: str = "", parent: QObject = None):
        super().__init__(parent)
        # Initialize flags
        self._wait_flag = False
        self._wait_msg = ""

        # Create graph objects
        self._config_file = config_file
        self._sgt_objs = {}
        self._selected_sgt_obj_index = 0

    @property
    def wait_flag(self) -> bool:
        """Returns the wait flag indicating if the application is currently running a task in the background."""
        return self._wait_flag

    @wait_flag.setter
    def wait_flag(self, value: bool):
        """Sets the wait flag indicating if the application is currently running a task in the background."""
        self._wait_flag = value

    @property
    def wait_msg(self) -> str:
        """Returns the wait message indicating the current task."""
        return self._wait_msg

    @wait_msg.setter
    def wait_msg(self, value: str):
        """Sets the wait message indicating the current task."""
        self._wait_msg = value

    @property
    def sgt_objs(self):
        return self._sgt_objs

    @property
    def selected_sgt_obj_index(self):
        return self._selected_sgt_obj_index

    @selected_sgt_obj_index.setter
    def selected_sgt_obj_index(self, value):
        self._selected_sgt_obj_index = value

    def replicate_sgt_configs(self) -> None:
        """Replicate the configurations of the selected SGT object to all other SGT objects."""
        # Update Configs
        current_sgt_obj = self.get_selected_sgt_obj()
        if current_sgt_obj is None:
            return

        keys_list = list(self._sgt_objs.keys())
        key_at_current = keys_list[self._selected_sgt_obj_index]
        shared_gtc_configs = current_sgt_obj.configs
        shared_gte_configs = current_sgt_obj.ntwk_p.graph_obj.configs
        shared_img_configs = current_sgt_obj.ntwk_p.image_obj.configs
        for key in keys_list:
            if key != key_at_current:
                s_obj = self._sgt_objs[key]
                s_obj.configs = shared_gtc_configs
                s_obj.ntwk_p.graph_obj.configs = shared_gte_configs
                for img_obj in s_obj.ntwk_p.selected_images:
                    img_obj.configs = shared_img_configs

    def get_selected_sgt_obj(self) -> GraphAnalyzer | None:
        """Retrieve the SGT object at a specified index."""
        try:
            keys_list = list(self._sgt_objs.keys())
            key_at_index = keys_list[self._selected_sgt_obj_index]
            sgt_obj = self._sgt_objs[key_at_index]
            return sgt_obj
        except IndexError:
            logging.info("No Image Error: Please import/add an image.", extra={'user': 'SGT Logs'})
            # self.showAlertSignal.emit("No Image Error", "No image added! Please import/add an image.")
            return None

    def get_selected_image(self, img_pos: int = 0, view: str = "original") -> str:
        """
        Finds image at a specific frame position and specified view (original or binary or processed or graph);
        then, returns it as a 'base64' string.

        Args:
            img_pos: Position index of the image-object in the selected batch.
            view: The current visualization type of the image (Original, Processed, Binary, Mutated, Graph).

        Returns:
            base64 string of the image-object.
        """
        try:
            ntwk_p = self.get_selected_sgt_obj().ntwk_p
            if view == "original":
                images = ntwk_p.image_3d
            elif view == "binary":
                images = ntwk_p.binary_image_3d
            elif view == "processed":
                images = ntwk_p.processed_image_3d
            elif view == "mutated":
                images = ntwk_p.mutated_image_3d
            elif view == "graph":
                images = [ntwk_p.graph_obj.img_ntwk]
            else:
                raise ValueError("View must be 'original', 'binary', 'processed', 'mutated', 'graph' or 'original'")

            if view == "graph":
                img_cv = images[0]
            else:
                img_cv = images[img_pos]

            if img_cv is None:
                raise ValueError(f"No image/graph found at position {img_pos}")

            b64_img = img_to_base64(img_cv)
            return b64_img
        except Exception as e:
            logging.error(f"Exception while getting selected image: {e}")
            return ""

    def get_selected_images(self):
        """
        Get selected images from a specific image batch.
        """
        sgt_obj = self.get_selected_sgt_obj()
        ntwk_p = sgt_obj.ntwk_p
        return ntwk_p.selected_images

    def update_sgt_obj(self, sgt_data: GraphAnalyzer|dict|None = None):
        """Update the SGT object at a specified index."""
        if sgt_data is None:
            return

        if isinstance(sgt_data, GraphAnalyzer):
            keys_list = list(self._sgt_objs.keys())
            key_at_index = keys_list[self._selected_sgt_obj_index]
            self._sgt_objs[key_at_index] = sgt_data

        if type(sgt_data) is dict:
            for key, obj in sgt_data.items():
                self._sgt_objs[key] = obj

    def create_sgt_object(self, file_path: str, out_dir: str = "") -> bool:
        """
        A function that processes a selected image file and creates an analyzer object with default configurations.

        Args:
            file_path: file path to the image or the graph data file.
            out_dir: output directory path

        Returns:
        """
        success, result = verify_path(file_path)
        if success:
            file_path = result
            success, result = verify_path(out_dir)
            out_dir = result if success else ""
        else:
            logging.info(result, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File Error", result)
            return False

        try:
            # Create an SGT object as a GraphAnalyzer object.
            file_ext = os.path.splitext(file_path)[1].lower()
            img_extensions = tuple(ext[1:] if ext.startswith('*.') else ext for ext in ALLOWED_IMG_EXTENSIONS)
            if file_ext.endswith(img_extensions):
                ntwk_p, file_name = ImageProcessor.from_image_file(file_path, out_folder=out_dir,
                                                              config_file=self._config_file, allow_auto_scale=True)
            else:
                ntwk_p, file_name = ImageProcessor.from_graph_file(file_path, out_folder=out_dir,)
            sgt_obj = GraphAnalyzer(ntwk_p)

            # Store the StructuralGT object and sync application
            self._sgt_objs[file_name] = sgt_obj
            return True
        except Exception as err:
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File Error", "Error processing image. Try again.")
            return False

    def delete_sgt_object(self, index=None) -> bool:
        """
        Delete SGT Obj stored at the specified index (if not specified, get the current index).
        """
        del_index = index if index is not None else self._selected_sgt_obj_index
        if 0 <= del_index < len(self._sgt_objs):  # Check if the index exists
            keys_list = list(self._sgt_objs.keys())
            key_at_del_index = keys_list[self._selected_sgt_obj_index]
            # Delete the object at index
            del self._sgt_objs[key_at_del_index]
            return True
        return False

    def update_output_dir(self, folder_path: str) -> None:
        """Update the output directory for storing StructuralGT results."""
        success, result = verify_path(folder_path)
        if success:
            folder_path = result
        else:
            try:
                os.makedirs(folder_path, exist_ok=True)
            except Exception as err:
                logging.exception("Folder Creation Error: %s", err, extra={'user': 'SGT Logs'})
                self.showAlertSignal.emit("Folder Creation Error", f"Error creating output folder: {err}")
                return

        # Update for all sgt_objs
        key_list = list(self._sgt_objs.keys())
        for key in key_list:
            sgt_obj = self._sgt_objs[key]
            sgt_obj.ntwk_p.output_dir = folder_path

    def add_single_image(self, img_path: str, out_dir: str = "") -> bool:
        """Verify and validate an image path, use it to create an SGT object and load it in view."""
        is_created = self.create_sgt_object(img_path, out_dir)
        if is_created:
            return True
        return False

    def add_multiple_images(self, img_dir_path: str, out_dir: str = "") -> bool:
        """
        Verify and validate multiple image paths, use each to create an SGT object, then load the last one in view.
        """
        success, result = verify_path(img_dir_path)
        if success:
            img_dir_path = result
        else:
            logging.info(result, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Directory Error", result)
            return False

        files = os.listdir(img_dir_path)
        files = sorted(files)
        for a_file in files:
            allowed_extensions = tuple(ext[1:] if ext.startswith('*.') else ext for ext in ALLOWED_IMG_EXTENSIONS)
            if a_file.endswith(allowed_extensions):
                img_path = os.path.join(str(img_dir_path), a_file)
                _ = self.create_sgt_object(img_path, out_dir)

        if len(self._sgt_objs) <= 0:
            logging.info("File Error: Files have to be either .tif .png .jpg .jpeg", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File Error",
                                      "No workable images found! Files have to be either .tif, .png, .jpg or .jpeg")
            return False
        else:
            return True

    def add_graph(self, graph_file: str, out_dir: str = "") -> bool:
        """Verify and validate the graph file path, use it to create an SGT object and load it in view."""
        is_created = self.create_sgt_object(graph_file, out_dir)
        if is_created:
            return True
        return False
