# SPDX-License-Identifier: GNU GPL v3

"""
Base worker class for executing all resource-intensive StructuralGT tasks.
"""

import logging
from PySide6.QtCore import QObject, Signal
from ...compute.graph_analyzer import GraphAnalyzer
from ...utils.sgt_utils import AbortException, plot_to_opencv, TaskResult, upload_to_dropbox, ProgressData


class BaseWorker:

    def __init__(self):
        self._progress_queue = None

    @property
    def progress_queue(self):
        return self._progress_queue

    def _update_progress(self, status_data: ProgressData):
        """
        Send the update_progress signal to all listeners.
        Progress-value (0-100), progress-message (str)
        Args:
            status_data: ProgressData object that contains the percentage and status message of the current task.

        Returns:

        """
        if self._progress_queue is None:
            return
        self._progress_queue.put(("progress", status_data))

    def attach_progress_queue(self, queue):
        """Attach or replace the progress queue (status_queue)."""
        if self._progress_queue is None:
            self._progress_queue = queue

    def task_save_images(self, ntwk_p, img_idx):
        """"""
        try:
            self._update_progress(ProgressData(percent=25, sender="GT", message=f"Saving Images..."))
            ntwk_p.save_images_to_file(img_pos=img_idx)
            self._update_progress(ProgressData(percent=95, sender="GT", message=f"Saving Images..."))
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
            self._update_progress(ProgressData(percent=25, sender="GT", message=f"Exporting Graph..."))
            filename, out_dir = ntwk_p.get_filenames()

            # 2. Save graph data to the file
            self._update_progress(ProgressData(percent=30, sender="GT", message=f"Exporting Graph..."))
            ntwk_p.graph_obj.save_graph_to_file(filename, out_dir)
            self._update_progress(ProgressData(percent=95, sender="GT", message=f"Exporting Graph..."))
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
            self._update_progress(ProgressData(type="error", sender="GT", message=f"Error encountered! Try again."))
            # Clean up listeners before exiting
            ntwk_p.remove_listener(self._update_progress)
            # Emit failure signal (aborted)
            return False, ["Apply Filters Failed", "Fatal error while applying filters! "
                                                                         "Change filter settings and try again; "
                                                                         "Or, Close the app and try again."]

    def task_calculate_img_histogram(self, ntwk_p, img_idx):
        """"""
        try:
            hist_images = ntwk_p.compute_img_histograms(img_pos=img_idx)
            if hist_images is None:
                self._update_progress(ProgressData(type="warning", sender="GT", message=f"Histogram calculation finished with failure"))
                return True, []

            self._update_progress(ProgressData(type="warning", sender="GT", message=f"Histogram calculation finished successfully"))
            return True, hist_images
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            return False, ["Histogram Calculation Failed", "Error while calculating image histogram!"]

    def task_retrieve_img_colors(self, ntwk_p, img_idx, max_colors=6):
        """"""
        def _generate_colors_data():
            """"""
            color_data = [{"id": i, "text": c.hex_code, "value": 1 if c.is_selected else 0} for i, c in enumerate(colors_found)]
            return color_data

        try:
            # ntwk_p.add_listener(self._update_progress)
            colors_found = ntwk_p.retrieve_dominant_img_colors(img_pos=img_idx, top_k=max_colors)
            # ntwk_p.remove_listener(self._update_progress)
            if colors_found is None:
                task_data = TaskResult(task_id="Image Colors", status="Finished", message="No dominant colors found!", data=[ntwk_p, None])
                return True, task_data

            ntwk_p.image_obj.dominant_colors = colors_found
            lst_colors = _generate_colors_data()
            task_data = TaskResult(task_id="Image Colors", status="Finished", message="Colors successfully retrieved!", data=[ntwk_p, lst_colors])
            return True, task_data
        except Exception as err:
            logging.exception(f"Color Error: {err}", extra={'user': 'SGT Logs'})
            self._update_progress(ProgressData(type="error", sender="GT", message=f"Error encountered! Try again."))
            # Clean up listeners before exiting
            # ntwk_p.remove_listener(self._update_progress)
            return False, ["Retrieve Colors Failed", "Error while retrieving image colors!"]

    def task_eliminate_img_colors(self, ntwk_p, img_idx, swap_color):
        """"""
        try:
            # ntwk_p.add_listener(self._update_progress)
            ntwk_p.eliminate_selected_img_colors(img_pos=img_idx, swap_color=swap_color)
            # ntwk_p.remove_listener(self._update_progress)
            task_data = TaskResult(task_id="Image Colors", status="Finished", message="Colors successfully eliminated!", data=[ntwk_p, None])
            return True, task_data
        except Exception as err:
            logging.exception(f"Color Error: {err}", extra={'user': 'SGT Logs'})
            self._update_progress(ProgressData(type="error", sender="GT", message=f"Error encountered! Try again."))
            # Clean up listeners before exiting
            # ntwk_p.remove_listener(self._update_progress)
            return False, ["Eliminate Colors Failed", "Error while eliminating image colors!"]

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
            task_data = TaskResult(task_id="Extract Graph", status="Finished", message="Graph extracted successfully!", data=ntwk_p)
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
            self._update_progress(ProgressData(type="error", sender="GT", message=f"Error encountered! Try again."))
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

    def task_rate_graph(self, score, ntwk_p):
        """Update score rating in graph properties and filter space results and upload graph image to DropBox App"""
        try:
            # 1. Convert the score from 1-10 to range 0-100
            is_successful = False
            percent_rating = score * 10
            ntwk_p.add_listener(self._update_progress)
            graph_file = ntwk_p.update_graph_rating(percent_rating)
            if graph_file is not None:
                # Upload image to DropBox App (in the future, to the server)
                _ = upload_to_dropbox(graph_file)
                self._update_progress(ProgressData(type="warning", sender="AI", message=f"Graph image uploaded to DropBox App!"))
                is_successful = True
            ntwk_p.remove_listener(self._update_progress)

            if is_successful:
                task_data = TaskResult(task_id="Rate Graph", status="Finished",
                                       message=f"Graph successfully rated {percent_rating}%.")
                return True, task_data
            else:
                return False, ["Graph Rating Failed", "Error occurred while rating graph!"]
        except Exception as err:
            logging.exception(err, extra={'user': 'SGT Logs'})
            ntwk_p.remove_listener(self._update_progress)
            return False, ["Graph Rating Aborted", "Error occurred while rating graph!"]


class BaseWorkerTerm(QObject, BaseWorker):

    inProgressSignal = Signal(object)

    def __init__(self):
        super(BaseWorkerTerm, self).__init__()

    def _update_progress(self, status_data: ProgressData):
        self.inProgressSignal.emit(status_data)


