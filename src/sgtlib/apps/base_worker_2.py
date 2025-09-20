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
            value: progress value (0-100), (-1, if it is an error), (101, if it is the nav-control message)
            msg: progress message (str)

        Returns:

        """
        self.progress_queue.put(("progress", (percent, msg)))

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