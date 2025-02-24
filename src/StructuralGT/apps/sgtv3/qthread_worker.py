import os
import time
import logging
from PySide6.QtCore import QThread,Signal

from src.StructuralGT.configs.config_loader import get_num_cores, write_file

class Worker(QThread):

    inProgressSignal = Signal(int, int, str)
    taskFinishedSignal = Signal(int, int, object)

    def __init__(self, func_id, args, target=None):
        super().__init__()
        self.__listeners = []
        self.target = target
        self.target_id = func_id
        self.args = args
        self.abort = False

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

    # Trigger events.
    def send_abort_message(self, args=None):
        self.abort = True
        if args is None:
            args = []
        for func in self.__listeners:
            func(*args)

    def run(self):
        if self.target:
            self.target(*self.args)
        if self.target_id == 0:
            self.service_filter_img(*self.args)
        elif self.target_id == 1:
            self.service_generate_graph(*self.args)
        elif self.target_id == 2:
            self.service_compute_gt(*self.args)
        elif self.target_id == 3:
            self.service_compute_gt_all(*self.args)
        elif self.target_id == 4:
            self.taskFinishedSignal.emit(3, 1, "Test complete!")

    def update_progress(self, value, msg):
        if value > 0:
            self.inProgressSignal.emit(value, value, msg)
        else:
            self.inProgressSignal.emit(0, value, msg)

    def service_filter_img(self, im_obj):
        try:
            self.update_progress(10, "applying filters...")
            im_obj.apply_filters()
            self.update_progress(100, '')
            self.taskFinishedSignal.emit(0, 0, im_obj)
            # graph_obj = GraphConverter(im_obj, options_gte=options_gte)
            # graph_obj.add_listener(self.update_progress)
            # graph_obj.fit_img()
            # graph_obj.remove_listener(self.update_progress)
            # self.taskFinishedSignal.emit(1, 0, graph_obj)
        except Exception as err:
            print(err)
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.abort = True
            self.update_progress(-1, "Error encountered! Try again")
            self.taskFinishedSignal.emit(0, 1, [])

    def service_generate_graph(self, graph_obj):
        try:
            graph_obj.abort = False
            graph_obj.add_listener(self.update_progress)
            self.add_thread_listener(graph_obj.abort_tasks)
            graph_obj.fit()
            graph_obj.remove_listener(self.update_progress)
            self.add_thread_listener(graph_obj.abort_tasks)
            self.taskFinishedSignal.emit(1, 0, graph_obj)
        except Exception as err:
            print(err)
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.abort = True
            self.update_progress(-1, "Error encountered! Try again")
            self.taskFinishedSignal.emit(1, 1, [])

    def service_compute_gt(self, sgt_obj):
        try:
            sgt_obj.g_obj.abort = False
            if sgt_obj.g_obj.nx_graph is None:
                sgt_obj.g_obj.add_listener(self.update_progress)
                self.add_thread_listener(sgt_obj.g_obj.abort_tasks)
                sgt_obj.g_obj.fit()
                sgt_obj.g_obj.remove_listener(self.update_progress)
                self.add_thread_listener(sgt_obj.g_obj.abort_tasks)
            else:
                if sgt_obj.g_obj.nx_graph.number_of_nodes() <= 0:
                    self.update_progress(-1, "Graph Error: Problem with graph (change/apply different filter and "
                                             "graph options). Or change brightness/contrast")
                    return
            if self.abort:
                self.update_progress(-1, "Task aborted.")
                self.taskFinishedSignal.emit(2, 1, [])
                return

            sgt_obj.add_listener(self.update_progress)
            self.add_thread_listener(sgt_obj.abort_tasks)
            sgt_obj.compute_gt_metrics()
            if sgt_obj.g_obj.configs.has_weights:
                if self.abort:
                    self.update_progress(-1, "Task aborted.")
                    self.taskFinishedSignal.emit(2, 1, [])
                    return
                sgt_obj.compute_weighted_gt_metrics()
            # metrics_obj.generate_pdf_output()
            if self.abort:
                self.update_progress(-1, "Task aborted.")
                self.taskFinishedSignal.emit(2, 1, [])
                return
            plot_figs = sgt_obj.generate_pdf_output(gui_app=True)
            sgt_obj.remove_listener(self.update_progress)
            self.remove_thread_listener(sgt_obj.abort_tasks)
            self.taskFinishedSignal.emit(2, 1, plot_figs)
        except Exception as err:
            print(err)
            logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
            self.abort = True
            self.update_progress(-1, "Error encountered! Try again")
            self.taskFinishedSignal.emit(2, 1, [])

    def service_compute_gt_all(self, sgt_objs):
        i = 0
        for sgt_obj in sgt_objs:
            start = time.time()

            if self.abort:
                self.update_progress(-1, "Task aborted.")
                self.taskFinishedSignal.emit(3, 0, [])
                return
            try:
                sgt_obj.g_obj.abort = False
                if sgt_obj.g_obj.nx_graph is None:
                    sgt_obj.g_obj.add_listener(self.update_progress)
                    sgt_obj.g_obj.fit()
                    sgt_obj.g_obj.remove_listener(self.update_progress)
                    self.add_thread_listener(sgt_obj.g_obj.abort_tasks)
                else:
                    if sgt_obj.g_obj.nx_graph.number_of_nodes() <= 0:
                        self.update_progress(-1, "Graph Error: Problem with graph (change/apply different filter and "
                                                 "graph options). Or change brightness/contrast")
                        return
                if self.abort:
                    self.update_progress(-1, "Task aborted.")
                    self.taskFinishedSignal.emit(3, 0, [])
                    return

                sgt_obj.add_listener(self.update_progress)
                self.add_thread_listener(sgt_obj.abort_tasks)
                sgt_obj.compute_gt_metrics()
                if sgt_obj.g_obj.configs.has_weights:
                    if self.abort:
                        self.update_progress(-1, "Task aborted.")
                        self.taskFinishedSignal.emit(3, 0, [])
                        return
                    sgt_obj.compute_weighted_gt_metrics()
                if self.abort:
                    self.update_progress(-1, "Task aborted.")
                    self.taskFinishedSignal.emit(3, 0, [])
                    return
                # metrics_obj.generate_pdf_output()
                plot_figs = sgt_obj.generate_pdf_output(gui_app=True)
                sgt_obj.remove_listener(self.update_progress)
                self.add_thread_listener(sgt_obj.abort_tasks)
                self.taskFinishedSignal.emit(2, 0, plot_figs)

                end = time.time()
                num_cores = get_num_cores()
                i += 1
                output = "Analyzing Image:" + str(i) + "/" + str(len(sgt_objs)) + "\n"
                output += "Run-time: " + str(end - start) + " seconds" + "\n"
                output += "Number of cores: " + str(num_cores) + "\n"
                output += "Results generated for: " + sgt_obj.g_obj.imp.img_path + "\n"
                output += "Node Count: " + str(sgt_obj.g_obj.nx_graph.number_of_nodes()) + "\n"
                output += "Edge Count: " + str(sgt_obj.g_obj.nx_graph.number_of_edges()) + "\n"
                filename, out_dir = sgt_obj.g_obj.imp.create_filenames()
                out_file = os.path.join(out_dir, filename + '-v2_results.txt')
                write_file(output, out_file)
                print(output)
                logging.info(output, extra={'user': 'SGT Logs'})
            except Exception as err:
                print(err)
                logging.exception("Error: %s", err, extra={'user': 'SGT Logs'})
                self.abort = True
                self.update_progress(-1, "Error encountered! Try again")
                self.taskFinishedSignal.emit(3, 0, [])
        self.taskFinishedSignal.emit(3, 1, [])
