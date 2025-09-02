# SPDX-License-Identifier: GNU GPL v3

"""
Base controller class for StructuralGT.
"""

import os
import logging
from PySide6.QtCore import Signal, QObject

from ..utils.sgt_utils import verify_path
from ..imaging.image_processor import ImageProcessor, ALLOWED_IMG_EXTENSIONS
from ..compute.graph_analyzer import GraphAnalyzer

class BaseController(QObject):

    showAlertSignal = Signal(str, str)

    def __init__(self, config_file: str = ""):
        super().__init__()
        # Create graph objects
        self._config_file = config_file
        self._sgt_objs = {}
        self._selected_sgt_obj_index = 0
        self._allow_auto_scale = True

    @property
    def sgt_objs(self):
        return self._sgt_objs

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
                                                              config_file=self._config_file,
                                                              allow_auto_scale=self._allow_auto_scale)
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
