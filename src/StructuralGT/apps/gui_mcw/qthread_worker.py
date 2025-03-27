import os
import time
import logging
from PySide6.QtCore import QObject,QThread,Signal


from ...SGT.sgt_utils import get_num_cores, write_txt_file


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
            im_obj.add_listener(self.update_progress)
            im_obj.apply_img_filters()
            im_obj.remove_listener(self.update_progress)
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

    def task_extract_graph(self, imp):
        """"""
        try:
            imp.abort = False
            imp.add_listener(self.update_progress)
            imp.apply_img_filters()
            imp.create_graphs()
            self.is_aborted(imp)
            imp.remove_listener(self.update_progress)
            self.taskFinishedSignal.emit(True, imp)
        except AbortException as err:
            print(f"Task aborted error: {err}")
            # Clean up listeners before exiting
            imp.remove_listener(self.update_progress)
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
            imp.remove_listener(self.update_progress)
            self.remove_thread_listener(imp.abort_tasks)
            # Emit failure signal (aborted)
            self.taskFinishedSignal.emit(False, ["Extract Graph Failed", "Graph extraction aborted due to error! "
                                                                          "Change image filters and/or graph settings "
                                                                          "and try again. If error persists then close "
                                                                          "the app and try again."])

    def task_compute_gt(self, sgt_obj):
        """"""
        success, result = self._compute_gt_parameters(sgt_obj)
        if success:
            self.taskFinishedSignal.emit(False, result)
        else:
            self.taskFinishedSignal.emit(False, ["SGT Computations Failed", "Fatal error occurred while computing "
                                                                            "GT parameters. Change image filters and/or "
                                                                            "graph settings and try again. If error "
                                                                            "persists then close the app and try again."])

    def task_compute_multi_gt(self, sgt_objs):
        """"""
        try:
            i = 0
            keys_list = list(sgt_objs.keys())
            for key in keys_list:
                sgt_obj = sgt_objs[key]

                status_msg = f"Analyzing Image: {(i + 1)} / {len(sgt_objs)}"
                self.update_progress(101, status_msg)

                start = time.time()
                success, result = self._compute_gt_parameters(sgt_obj)
                self.is_aborted(sgt_obj)
                self.taskFinishedSignal.emit(False, result)
                end = time.time()

                i += 1
                num_cores = get_num_cores()
                output = status_msg + "\n" + f"Run-time: {str(end - start)}  seconds\n"
                output += "Number of cores: " + str(num_cores) + "\n"
                output += "Results generated for: " + sgt_obj.imp.img_path + "\n"
                output += "Node Count: " + str(sgt_obj.g_obj.nx_graph.number_of_nodes()) + "\n"
                output += "Edge Count: " + str(sgt_obj.g_obj.nx_graph.number_of_edges()) + "\n"
                filename, out_dir = sgt_obj.imp.create_filenames()
                out_file = os.path.join(out_dir, filename + '-v2_results.txt')
                write_txt_file(output, out_file)
                print(output)
                logging.info(output, extra={'user': 'SGT Logs'})
            self.taskFinishedSignal.emit(True, sgt_objs)
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

    def _compute_gt_parameters(self, sgt_obj):
        """"""
        try:
            # Add Listeners
            sgt_obj.add_listener(self.update_progress)
            self.add_thread_listener(sgt_obj.abort_tasks)

            sgt_obj.run_analyzer()
            self.is_aborted(sgt_obj)

            # Cleanup - remove listeners
            sgt_obj.remove_listener(self.update_progress)
            self.remove_thread_listener(sgt_obj.abort_tasks)
            return True, sgt_obj
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
    def is_aborted(active_obj):
        """Raise an exception if the process is aborted."""
        if active_obj.abort:
            raise AbortException("Process aborted")
