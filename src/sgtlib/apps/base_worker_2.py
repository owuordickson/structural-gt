# SPDX-License-Identifier: GNU GPL v3

"""
Base worker class for executing all resource-intensive StructuralGT tasks.
"""

import logging
from ..compute.graph_analyzer import GraphAnalyzer
from ..utils.sgt_utils import AbortException, plot_to_opencv, TaskResult


class BaseWorker:

    def __init__(self):
        self.progress_queue = None

    def _update_progress(self, percent, msg):
        """
        Send the update_progress signal to all listeners.
        Progress-value (0-100), progress-message (str)
        Args:
            percent: progress value (0-100), (-1, if it is an error), (101, if it is the nav-control message)
            msg: progress message (str)

        Returns:

        """
        if self.progress_queue is None:
            return
        self.progress_queue.put(("progress", (percent, msg)))

    def task_save_images(self, ntwk_p):
        """"""
        try:
            self._update_progress(25, "Saving Images...")
            ntwk_p.save_images_to_file()
            self._update_progress(95, "Saving Images...")
            task_data = TaskResult(task_id="Save Images", status="Finished",
                                   message="Image files successfully saved in 'Output Dir'")
            return True, task_data
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            return False, ["Save Images Failed", "Error while saving images!"]

    def task_export_graph(self, ntwk_p):
        """"""
        try:
            # 1. Get filename
            self._update_progress(25, "Exporting Graph...")
            filename, out_dir = ntwk_p.get_filenames()

            # 2. Save graph data to the file
            self._update_progress(30, "Exporting Graph...")
            ntwk_p.graph_obj.save_graph_to_file(filename, out_dir)
            self._update_progress(95, "Exporting Graph...")
            task_data = TaskResult(task_id="Export Graph", status="Finished",
                                   message="Graph successfully exported to file and saved in 'Output Dir'")
            return True, task_data
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            return False, ["Export Graph Failed", "Error while exporting graph!"]

    def task_apply_img_filters(self, ntwk_p):
        """"""
        try:
            ntwk_p.add_listener(self._update_progress)
            ntwk_p.apply_img_filters()
            ntwk_p.remove_listener(self._update_progress)
            task_data = TaskResult(task_id="Apply Filters", status="Finished", data=ntwk_p)
            return True, task_data
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            # self.abort = True
            self._update_progress(-1, "Error encountered! Try again")
            # Emit failure signal (aborted)
            return False, ["Apply Filters Failed", "Fatal error while applying filters! "
                                                                         "Change filter settings and try again; "
                                                                         "Or, Close the app and try again."]

    def task_calculate_img_histogram(self, ntwk_p):
        """"""
        try:
            hist_images = [plot_to_opencv(obj.plot_img_histogram(curr_view=ntwk_p.selected_batch_view)) for obj in ntwk_p.image_obj_3d]
            self._update_progress(101, "Histogram Calculation Finished")
            return True, hist_images
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            return False, ["Histogram Calculation Failed", "Error while calculating image histogram!"]

    def task_extract_graph(self, ntwk_p):
        """"""
        try:
            ntwk_p.abort = False
            ntwk_p.add_listener(self._update_progress)
            ntwk_p.apply_img_filters()
            ntwk_p.build_graph_network()
            if ntwk_p.abort:
                raise AbortException("Process aborted")
            ntwk_p.remove_listener(self._update_progress)
            task_data = TaskResult(task_id="Extract Graph", status="Finished", message="", data=ntwk_p)
            return True, task_data
        except AbortException as err:
            logging.exception("Task Aborted: %s", err, extra={'user': 'SGT Logs'})
            # Clean up listeners before exiting
            ntwk_p.remove_listener(self._update_progress)
            return False, ["Extract Graph Aborted", "Graph extraction aborted due to error! "
                                                                          "Change image filters and/or graph settings "
                                                                          "and try again. If error persists then close "
                                                                          "the app and try again."]
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self._update_progress(-1, "Error encountered! Try again")
            # Clean up listeners before exiting
            ntwk_p.remove_listener(self._update_progress)
            # Emit failure signal (aborted)
            return False, ["Extract Graph Failed", "Graph extraction aborted due to error! "
                                                                          "Change image filters and/or graph settings "
                                                                          "and try again. If error persists then close "
                                                                          "the app and try again."]

    def task_compute_gt(self, sgt_obj):
        """"""
        success, new_sgt = GraphAnalyzer.safe_run_analyzer(sgt_obj, self._update_progress, save_to_pdf=True)
        if success:
            task_data = TaskResult(task_id="Compute GT", status="Finished", data=new_sgt)
            return True, task_data
        else:
            return False, ["SGT Computations Failed", "Fatal error occurred while computing GT parameters. Change image filters and/or graph settings and try again. If error persists then close the app and try again."]

    def task_compute_multi_gt(self, sgt_objs):
        """"""
        new_sgt_objs = GraphAnalyzer.safe_run_multi_analyzer(sgt_objs, self._update_progress)
        if new_sgt_objs is not None:
            task_data = TaskResult(task_id="Compute Multi GT", status="Finished", data=new_sgt_objs)
            return True, task_data
        else:
            msg = "Either task was aborted by user or a fatal error occurred while computing GT parameters. Change image filters and/or graph settings and try again. If error persists then close the app and try again."
            return False, ["SGT Computations Aborted/Failed", msg]

    def task_metaheuristic_search(self, ntwk_p):
        """"""
        try:
            if ntwk_p.filter_space is not None:
                if ntwk_p.filter_space.best_candidate.position not in ntwk_p.filter_space.ignore_candidates:
                    # Filters already selected and values estimated
                    task_data = TaskResult(task_id="Metaheuristic Search", status="Stopped", data=ntwk_p)
                    return True, task_data

            ntwk_p.abort = False
            ntwk_p.add_listener(self._update_progress)
            img_configs = ntwk_p.metaheuristic_image_configs()
            if ntwk_p.abort:
                raise AbortException("Task stopped")
            ntwk_p.image_obj.configs = img_configs.copy()
            ntwk_p.remove_listener(self._update_progress)
            task_data = TaskResult(task_id="Metaheuristic Search", status="Finished", data=ntwk_p)
            return True, task_data
        except AbortException as err:
            logging.exception("Task Stopped: %s", err, extra={'user': 'SGT Logs'})
            ntwk_p.remove_listener(self._update_progress)
            return False, ["Metaheuristic Search Stopped", "Search stopped by user or due to error!"]
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            ntwk_p.remove_listener(self._update_progress)
            return False, ["Metaheuristic Search Failed", "Error occurred while running metaheuristic search!"]
