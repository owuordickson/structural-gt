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
            TerminalApp.is_aborted(ntwk_p)
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
        success, new_sgt = TerminalApp.compute_gt_parameters(sgt_obj)
        if success:
            GraphAnalyzer.write_to_pdf(new_sgt, TerminalApp.update_progress)
            return new_sgt
        else:
            msg = "Fatal error occurred while computing GT parameters. Change image filters and/or  graph settings and try again. If error persists then close the app and try again."
            logging.info(f"SGT Computations Failed: {msg}", extra={'user': 'SGT Logs'})
            return None

    def task_compute_multi_gt(self, sgt_objs):
        """"""
        try:
            i = 0
            keys_list = list(sgt_objs.keys())
            for key in keys_list:
                sgt_obj = sgt_objs[key]

                status_msg = f"Analyzing Image: {(i + 1)} / {len(sgt_objs)}"
                TerminalApp.update_progress(101, status_msg)

                start = time.time()
                success, new_sgt = TerminalApp.compute_gt_parameters(sgt_obj)
                TerminalApp.is_aborted(sgt_obj)
                if success:
                    GraphAnalyzer.write_to_pdf(new_sgt, TerminalApp.update_progress)
                end = time.time()

                i += 1
                num_cores = get_num_cores()
                sel_batch = sgt_obj.ntwk_p.get_selected_batch()
                graph_obj = sel_batch.graph_obj
                output = status_msg + "\n" + f"Run-time: {str(end - start)}  seconds\n"
                output += "Number of cores: " + str(num_cores) + "\n"
                output += "Results generated for: " + sgt_obj.ntwk_p.img_path + "\n"
                output += "Node Count: " + str(graph_obj.nx_3d_graph.number_of_nodes()) + "\n"
                output += "Edge Count: " + str(graph_obj.nx_3d_graph.number_of_edges()) + "\n"
                filename, out_dir = sgt_obj.ntwk_p.get_filenames()
                out_file = os.path.join(out_dir, filename + '-v2_results.txt')
                write_txt_file(output, out_file)
                logging.info(output, extra={'user': 'SGT Logs'})
            return sgt_objs
        except AbortException as err:
            logging.exception("Task Aborted: %s", err, extra={'user': 'SGT Logs'})
            TerminalApp.update_progress(-1, "All tasks aborted!")
            # Emit failure signal (aborted)
            msg = "Graph theory parameter computations aborted by user."
            logging.info(f"SGT Computations Aborted: {msg}", extra={'user': 'SGT Logs'})
            return None
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            TerminalApp.update_progress(-1, "Error encountered! Try again")
            # Emit failure signal (aborted)
            msg = "Fatal error occurred while computing GT parameters. Change image filters and/or  graph settings and try again. If error persists then close the app and try again."
            logging.info(f"SGT Computations Failed: {msg}", extra={'user': 'SGT Logs'})
            return None

    @staticmethod
    def compute_gt_parameters(sgt_obj):
        """"""
        try:
            # Add Listeners
            sgt_obj.add_listener(TerminalApp.update_progress)

            sgt_obj.run_analyzer()
            TerminalApp.is_aborted(sgt_obj)

            # Cleanup - remove listeners
            sgt_obj.remove_listener(TerminalApp.update_progress)
            return True, sgt_obj
        except AbortException as err:
            logging.exception("Task Aborted: %s", err, extra={'user': 'SGT Logs'})
            TerminalApp.update_progress(-1, "Task aborted!")
            # Clean up listeners before exiting
            sgt_obj.remove_listener(TerminalApp.update_progress)
            return False, None
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            TerminalApp.update_progress(-1, "Error encountered! Try again")
            # Clean up listeners before exiting
            sgt_obj.remove_listener(TerminalApp.update_progress)
            return False, None

    @staticmethod
    def is_aborted(active_obj):
        """Raise an exception if the process is aborted."""
        if active_obj.abort:
            raise AbortException("Process aborted")

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

    is_multi = configs.multiImage
    img_path = configs.filePath
    out_dir = configs.outputDir
    filenames = []
