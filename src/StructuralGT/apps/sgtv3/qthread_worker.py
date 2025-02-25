import os
import time
import logging
from PySide6.QtCore import QObject,QThread,Signal

from src.StructuralGT.configs.config_loader import get_num_cores, write_file


class AbortException(Exception):
    """Custom exception to handle task abortion."""
    pass


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
    # errorOccurredSignal = Signal()
    # abortSignal = Signal()

    def __init__(self):
        super().__init__()
        self.__listeners = []

    def add_thread_listener(self, func):
        """
        Add functions from the list of listeners.
        :param func:
        :return:
        """
        if func in self.__listeners:
            return
        self.__listeners.append(func)

    def remove_thread_listener(self, func):
        """
        Remove functions from the list of listeners.
        :param func:
        :return:
        """
        if func not in self.__listeners:
            return
        self.__listeners.remove(func)

    def send_abort_signal(self, args=None):
        if args is None:
            args = []
        for func in self.__listeners:
            func(*args)

    def update_progress(self, value, msg):
        """
        Send update_progress signal to all listeners.
        progress-value (0-100), progress-message (str)
        Args:
            value: progress value (0-100), (-1, if it is an error), (101, if it is nav-control message)
            msg: progress message (str)

        Returns:

        """
        self.inProgressSignal.emit(value, msg)

    def task_apply_img_filters(self, im_obj):
        """"""
        try:
            self.update_progress(10, "applying filters...")
            im_obj.apply_filters()
            self.update_progress(100, '')
            self.taskFinishedSignal.emit(True, im_obj)
        except Exception as err:
            print(err)
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            # self.abort = True
            self.update_progress(-1, "Error encountered! Try again")
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["Apply Filters Failed", "Fatal error while applying filters! "
                                                                         "Change filter settings and try again; "
                                                                         "Or, Close the app and try again."])

    def task_extract_graph(self, graph_obj):
        """"""
        try:
            # graph_obj.abort = False
            graph_obj.add_listener(self.update_progress)
            graph_obj.fit()
            self.is_aborted(graph_obj)
            graph_obj.remove_listener(self.update_progress)
            self.taskFinishedSignal.emit(True, graph_obj)
        except AbortException as err:
            print(f"Task aborted error: {err}")
            # Clean up listeners before exiting
            graph_obj.remove_listener(self.update_progress)
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["Extract Graph Aborted", "Graph extraction aborted due to error! "
                                                                          "Change image filters and/or graph settings "
                                                                          "and try again. If error persists then close "
                                                                          "the app and try again."])
        except Exception as err:
            print(err)
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.update_progress(-1, "Error encountered! Try again")
            # Clean up listeners before exiting
            graph_obj.remove_listener(self.update_progress)
            self.remove_thread_listener(graph_obj.abort_tasks)
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["Extract Graph Failed", "Graph extraction aborted due to error! "
                                                                          "Change image filters and/or graph settings "
                                                                          "and try again. If error persists then close "
                                                                          "the app and try again."])

    def task_compute_gt(self, sgt_obj):
        """"""
        success, result = self.compute_gt_parameters(sgt_obj)
        self.taskFinishedSignal.emit(success, result)

    def task_compute_gt_all(self, sgt_objs):
        """"""
        try:
            i = 0
            for sgt_obj in sgt_objs:
                status_msg = f"Analyzing Image: {(i + 1)} / {len(sgt_objs)}"
                self.update_progress(101, status_msg)

                start = time.time()
                success, result = self.compute_gt_parameters(sgt_obj)
                self.is_aborted(sgt_obj)
                self.taskFinishedSignal.emit(False, result)
                end = time.time()

                i += 1
                num_cores = get_num_cores()
                output = status_msg + "\n" + f"Run-time: {str(end - start)}  seconds\n"
                output += "Number of cores: " + str(num_cores) + "\n"
                output += "Results generated for: " + sgt_obj.g_obj.imp.img_path + "\n"
                output += "Node Count: " + str(sgt_obj.g_obj.nx_graph.number_of_nodes()) + "\n"
                output += "Edge Count: " + str(sgt_obj.g_obj.nx_graph.number_of_edges()) + "\n"
                filename, out_dir = sgt_obj.g_obj.imp.create_filenames()
                out_file = os.path.join(out_dir, filename + '-v2_results.txt')
                write_file(output, out_file)
                print(output)
                logging.info(output, extra={'user': 'SGT Logs'})
            self.taskFinishedSignal.emit(True, None)
        except AbortException as err:
            print(f"All tasks aborted: {err}")
            self.update_progress(-1, "All tasks aborted!")
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["SGT Computations Aborted", "Graph theory parameter computations "
                                                                             "aborted by user."])
        except Exception as err:
            print(f"Error:  {err}")
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.update_progress(-1, "Error encountered! Try again")
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["SGT Computations Failed", "Fatal error occurred while computing "
                                                                            "GT parameters. Change image filters and/or "
                                                                            "graph settings and try again. If error "
                                                                            "persists then close the app and try again."])

    def compute_gt_parameters(self, sgt_obj):
        """"""
        try:
            # Add Listeners
            sgt_obj.add_listener(self.update_progress)
            self.add_thread_listener(sgt_obj.abort_tasks)

            # 1. Apply image filters and extract graph
            sgt_obj.fit()
            self.is_aborted(sgt_obj)  # Check if function aborted

            # 2. Compute GT parameters
            sgt_obj.compute_gt_metrics()
            if sgt_obj.g_obj.configs["has_weights"]["value"]:
                self.is_aborted(sgt_obj)
                # 3. Compute weighted-GT parameters
                sgt_obj.compute_weighted_gt_metrics()
            self.is_aborted(sgt_obj)

            # 4. Generate results PDF
            plot_figs = sgt_obj.generate_pdf_output(gui_app=True)

            # Cleanup - remove listeners
            sgt_obj.remove_listener(self.update_progress)
            self.remove_thread_listener(sgt_obj.abort_tasks)
            return True, plot_figs
        except AbortException as err:
            print(f"Task aborted: {err}")
            self.update_progress(-1, "Task aborted!")
            # Clean up listeners before exiting
            sgt_obj.remove_listener(self.update_progress)
            self.remove_thread_listener(sgt_obj.abort_tasks)
            return False, None
        except Exception as err:
            print(f"Error:  {err}")
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.update_progress(-1, "Error encountered! Try again")
            # Clean up listeners before exiting
            sgt_obj.remove_listener(self.update_progress)
            self.remove_thread_listener(sgt_obj.abort_tasks)
            return False, None

    @staticmethod
    def is_aborted(sgt_obj):
        """Raise an exception if the process is aborted."""
        if sgt_obj.abort:
            raise AbortException("Process aborted")
