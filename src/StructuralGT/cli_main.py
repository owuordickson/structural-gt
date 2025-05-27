# SPDX-License-Identifier: GNU GPL v3

"""
Terminal interface implementations
"""

import os
import time
import logging

from src.StructuralGT import ALLOWED_IMG_EXTENSIONS
from src.StructuralGT.utils.sgt_utils import get_num_cores, verify_path, AbortException, write_txt_file
from src.StructuralGT.imaging.image_processor import ImageProcessor, FiberNetworkBuilder
from src.StructuralGT.compute.graph_analyzer import GraphAnalyzer



class TerminalApp:
    """Exposes the terminal interface for StructuralGT."""

    def __init__(self):
        # Create graph objects
        self.allow_auto_scale = True
        self.sgt_objs = {}

    def create_sgt_object(self, img_path):
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
            return False

        # Create an SGT object as a GraphAnalyzer object.
        try:
            ntwk_p, img_file = ImageProcessor.create_imp_object(img_path, self.allow_auto_scale)
            sgt_obj = GraphAnalyzer(ntwk_p)
            self.sgt_objs[img_file] = sgt_obj
            return True
        except Exception as err:
            logging.exception("File Error: %s", err, extra={'user': 'SGT Logs'})
            return False

    def add_single_image(self, image_path):
        """
        Verify and validate an image path, use it to create an SGT object

        :param image_path: image path to be processed
        :return: bool result of SGT object creation
        """
        is_created = self.create_sgt_object(image_path)
        if not is_created:
            logging.info("Fatal Error: Unable to create SGT object", extra={'user': 'SGT Logs'})
        return is_created

    def add_multiple_images(self, img_dir_path):
        """
        Verify and validate multiple image paths, use each to create an SGT object.
        """

        success, result = verify_path(img_dir_path)
        if success:
            img_dir_path = result
        else:
            logging.info(result, extra={'user': 'SGT Logs'})
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
            return False
        else:
            return True

    def task_extract_graph(self, ntwk_p):
        """"""
        try:
            ntwk_p.abort = False
            ntwk_p.add_listener(TerminalApp.update_progress)
            ntwk_p.apply_img_filters()
            ntwk_p.build_graph_network()
            if ntwk_p.abort:
                raise AbortException("Process aborted")
            ntwk_p.remove_listener(TerminalApp.update_progress)
            TerminalApp.update_progress(100, "Graph successfully extracted!")
            return ntwk_p
        except AbortException as err:
            logging.exception("Task Aborted: %s", err, extra={'user': 'SGT Logs'})
            # Clean up listeners before exiting
            ntwk_p.remove_listener(TerminalApp.update_progress)
            # Emit failure signal (aborted)
            msg = "Graph extraction aborted due to error! Change image filters and/or graph settings and try again. If error persists then close the app and try again"
            logging.info(f"Extract Graph Aborted: {msg}", extra={'user': 'SGT Logs'})
            return None

    def task_compute_gt(self, sgt_obj):
        """"""
        success, new_sgt = GraphAnalyzer.safe_run_analyzer(sgt_obj, TerminalApp.update_progress)
        if success:
            GraphAnalyzer.write_to_pdf(new_sgt, TerminalApp.update_progress)
            return new_sgt
        else:
            msg = "Either task was aborted by user or a fatal error occurred while computing GT parameters. Change image filters and/or graph settings and try again. If error persists then close the app and try again."
            logging.info(f"SGT Computations Failed: {msg}", extra={'user': 'SGT Logs'})
            return None

    def task_compute_multi_gt(self, sgt_objs):
        """"""
        new_sgt_objs = GraphAnalyzer.safe_run_multi_analyzer(sgt_objs, TerminalApp.update_progress)
        if new_sgt_objs is None:
            msg = "Either task was aborted by user or a fatal error occurred while computing GT parameters. Change image filters and/or graph settings and try again. If error persists then close the app and try again."
            logging.info(f"SGT Computations Failed: {msg}", extra={'user': 'SGT Logs'})
        return new_sgt_objs

    @staticmethod
    def update_progress(value, msg):
        """
        Simple method to display progress updates.

        Args:
            value (int): progress value
            msg (str): progress message
        Returns:
             None:
        """
        print(str(value) + "%: " + msg)
        logging.info(str(value) + "%: " + msg, extra={'user': 'SGT Logs'})



def terminal_app():
    """
    Initializes and executes StructuralGT functions.
    :return:
    """
    configs = {}  # load_project_configs()

    """is_multi = configs.multiImage
    img_path = configs.filePath
    out_dir = configs.outputDir
    filenames = []"""
