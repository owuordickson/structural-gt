import logging
from PySide6.QtCore import QObject, QThread, Signal
from ...compute.graph_analyzer import GraphAnalyzer
from ...utils.sgt_utils import AbortException, plot_to_opencv



class QThreadWorker(QThread):
    def __init__(self, func, args, parent=None):
        super().__init__(parent)
        self.func = func  # Store function reference
        self.args = args  # Store arguments

    def run(self):
        if self.func:
            self.func(*self.args)  # Call function with arguments


class WorkerTask (QObject):

    inProgressSignal = Signal(int, str)         # progress-value (0-100), progress-message (str)
    taskFinishedSignal = Signal(bool, object)    # success/fail (True/False), result (object)

    def __init__(self):
        super().__init__()

    def update_progress(self, value, msg):
        """
        Send the update_progress signal to all listeners.
        Progress-value (0-100), progress-message (str)
        Args:
            value: progress value (0-100), (-1, if it is an error), (101, if it is the nav-control message)
            msg: progress message (str)

        Returns:

        """
        self.inProgressSignal.emit(value, msg)

    def task_apply_img_filters(self, ntwk_p):
        """"""
        try:
            ntwk_p.add_listener(self.update_progress)
            ntwk_p.apply_img_filters()
            ntwk_p.remove_listener(self.update_progress)
            self.taskFinishedSignal.emit(True, ntwk_p)
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            # self.abort = True
            self.update_progress(-1, "Error encountered! Try again")
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["Apply Filters Failed", "Fatal error while applying filters! "
                                                                         "Change filter settings and try again; "
                                                                         "Or, Close the app and try again."])

    def task_calculate_img_histogram(self, ntwk_p):
        """"""
        try:
            sel_img_batch = ntwk_p.get_selected_batch()
            hist_images = [plot_to_opencv(obj.plot_img_histogram(curr_view=sel_img_batch.current_view)) for obj in ntwk_p.image_obj_3d]
            self.taskFinishedSignal.emit(True, hist_images)
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.taskFinishedSignal.emit(False, ["Histogram Calculation Failed", "Error while calculating image histogram!"])

    def task_extract_graph(self, ntwk_p):
        """"""
        try:
            ntwk_p.abort = False
            ntwk_p.add_listener(self.update_progress)
            ntwk_p.apply_img_filters()
            ntwk_p.build_graph_network()
            if ntwk_p.abort:
                raise AbortException("Process aborted")
            ntwk_p.remove_listener(self.update_progress)
            self.taskFinishedSignal.emit(True, ntwk_p)
        except AbortException as err:
            logging.exception("Task Aborted: %s", err, extra={'user': 'SGT Logs'})
            # Clean up listeners before exiting
            ntwk_p.remove_listener(self.update_progress)
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["Extract Graph Aborted", "Graph extraction aborted due to error! "
                                                                          "Change image filters and/or graph settings "
                                                                          "and try again. If error persists then close "
                                                                          "the app and try again."])
        except Exception as err:
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.update_progress(-1, "Error encountered! Try again")
            # Clean up listeners before exiting
            ntwk_p.remove_listener(self.update_progress)
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["Extract Graph Failed", "Graph extraction aborted due to error! "
                                                                          "Change image filters and/or graph settings "
                                                                          "and try again. If error persists then close "
                                                                          "the app and try again."])

    def task_compute_gt(self, sgt_obj):
        """"""
        success, new_sgt = GraphAnalyzer.safe_run_analyzer(sgt_obj, self.update_progress, save_to_pdf=True)
        if success:
            self.taskFinishedSignal.emit(True, new_sgt)
        else:
            self.taskFinishedSignal.emit(False, ["SGT Computations Failed", "Fatal error occurred while computing GT parameters. Change image filters and/or graph settings and try again. If error persists then close the app and try again."])

    def task_compute_multi_gt(self, sgt_objs):
        """"""
        new_sgt_objs = GraphAnalyzer.safe_run_multi_analyzer(sgt_objs, self.update_progress)
        if new_sgt_objs is not None:
            self.taskFinishedSignal.emit(True, sgt_objs)
        else:
            msg = "Either task was aborted by user or a fatal error occurred while computing GT parameters. Change image filters and/or graph settings and try again. If error persists then close the app and try again."
            self.taskFinishedSignal.emit(False, ["SGT Computations Aborted/Failed", msg])
