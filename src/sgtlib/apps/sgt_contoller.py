# SPDX-License-Identifier: GNU GPL v3

"""
Base controller class for StructuralGT.
"""

import os
import sys
import logging
from PySide6.QtCore import Signal

from ..utils.sgt_utils import verify_path
from ..imaging.image_processor import ImageProcessor, ALLOWED_IMG_EXTENSIONS
from ..compute.graph_analyzer import GraphAnalyzer

class BaseController:

    showAlertSignal = Signal(str, str)

    def __init__(self):
        # Create graph objects
        self.sgt_objs = {}
        self.selected_sgt_obj_index = 0
        self.allow_auto_scale = True

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

    def create_sgt_object(self, img_path: str) -> bool:
        """
        A function that processes a selected image file and creates an analyzer object with default configurations.

        Args:
            img_path: file path to image

        Returns:
        """
        success, result = verify_path(img_path)
        if success:
            img_path = result
        else:
            logging.info(result, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File/Directory Error", result)
            return False

        # Create an SGT object as a GraphAnalyzer object.
        try:
            ntwk_p, img_file = ImageProcessor.create_imp_object(img_path, config_file="", allow_auto_scale=self.allow_auto_scale)
            sgt_obj = GraphAnalyzer(ntwk_p)

            # Store the StructuralGT object and sync application
            self.sgt_objs[img_file] = sgt_obj
            return True
        except Exception as err:
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File Error", "Error processing image. Try again.")
            return False

    def update_output_dir(self, folder_path: str) -> None:
        """Update the output directory for storing StructuralGT results."""
        # Convert QML "file:///" path format to a proper OS path
        if folder_path.startswith("file:///"):
            if sys.platform.startswith("win"):  # Windows Fix (remove extra '/')
                folder_path = folder_path[8:]
            else:  # macOS/Linux (remove "file://")
                folder_path = folder_path[7:]
        folder_path = os.path.normpath(folder_path)  # Normalize the path

        # Update for all sgt_objs
        key_list = list(self.sgt_objs.keys())
        for key in key_list:
            sgt_obj = self.sgt_objs[key]
            sgt_obj.ntwk_p.output_dir = folder_path

    def add_single_image(self, image_path: str) -> bool:
        """Verify and validate an image path, use it to create an SGT object and load it in view."""
        is_created = self.create_sgt_object(image_path)
        if is_created:
            return True
        return False

    def add_multiple_images(self, img_dir_path: str) -> bool:
        """
        Verify and validate multiple image paths, use each to create an SGT object, then load the last one in view.
        """
        success, result = verify_path(img_dir_path)
        if success:
            img_dir_path = result
        else:
            logging.info(result, extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("File/Directory Error", result)
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
            self.showAlertSignal.emit("File Error",
                                      "No workable images found! Files have to be either .tif, .png, .jpg or .jpeg")
            return False
        else:
            return True