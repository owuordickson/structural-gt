# SPDX-License-Identifier: GNU GPL v3

"""
Terminal interface implementations
"""

import sys
import logging
from optparse import OptionParser

from .workers.base_workers import BaseWorkerTerm
from ..utils.sgt_utils import TaskResult, ProgressData
from .controllers.base_controller import BaseController
from ..utils.config_loader import strict_read_config_file

logger = logging.getLogger("SGT App")
#logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", stream=sys.stdout)

class TerminalApp(BaseController):
    """Exposes the terminal interface for StructuralGT."""

    def __init__(self, config_path: str):
        """
        Exposes methods for running StructuralGT tasks
        :param config_path: the path to the configuration file
        """
        super().__init__(config_file=config_path)
        self._task_worker = BaseWorkerTerm()
        self._task_worker.inProgressSignal.connect(TerminalApp.update_progress)
        self._selected_sgt_obj_index = 0
        # self.showAlertSignal.connect(self.show_alert)

    @property
    def task_worker(self):
        return self._task_worker

    def check_image_files(self, img_path: str="", img_dir: str = "", out_dir: str = ""):
        """"""
        # 1. Verify config file
        config_file_ok = strict_read_config_file(self._config_file, self.update_progress)
        if not config_file_ok:
            sys.exit('Usage: StructuralGT-cli -f datasets/InVitroBioFilm.png -c sgt_configs.ini -t 2 -o results/')

        # 2. Get images and process them
        if img_path != "":
            self.add_single_image(img_path, out_dir)
        elif img_dir != "":
            self.add_multiple_images(img_dir, out_dir)
        else:
            self.update_progress(ProgressData(type="error", sender="GT", message=f"No image(s) found in the path/image folder provided!"))
            sys.exit('System exit')

    @staticmethod
    def task_finished(success_val: bool, result: None | list | TaskResult) -> None:
        """
        Handler function for sending updates/signals on termination of tasks.
        Args:
            success_val:
            result:

        Returns:

        """
        if not success_val:
            if type(result) is list:
                logging.info(result[0] + ": " + result[1], extra={'user': 'SGT Logs'})
                TerminalApp.update_progress(ProgressData(type="info", sender="GT", message=f"{result[0]}: {result[1]}"))
        else:
            if isinstance(result, TaskResult):
                if result.task_id == "Apply Filters":
                    TerminalApp.update_progress(ProgressData(percent=100, sender="GT", message=f"Filters applied successfully!"))
                if result.task_id == "Export Graph" or result.task_id == "Save Images":
                    TerminalApp.update_progress(ProgressData(percent=100, sender="GT", message=f"Files Saved!"))
                if result.task_id == "Extract Graph":
                    TerminalApp.update_progress(ProgressData(percent=100, sender="GT", message=f"Graph extracted successfully!"))
                if result.task_id == "Compute GT":
                    TerminalApp.update_progress(ProgressData(percent=100, sender="GT", message=f"GT PDF successfully generated! Check it out in 'Output Dir'."))
                if result.task_id == "Compute Multi GT":
                    TerminalApp.update_progress(ProgressData(percent=100, sender="GT", message=f"All GT PDF successfully generated! Check it out in 'Output Dir'."))
                if result.task_id == "Metaheuristic Search":
                    if result.status == "Finished":
                        TerminalApp.update_progress(ProgressData(percent=100, sender="AI", message=f"Search completed!"))
            elif type(result) is list:
                # Histogram data
                pass
            else:
                pass

    @staticmethod
    def update_progress(status_data: ProgressData) -> None:
        """
        Simple method to display progress updates.

        Args:
            status_data: ProgressData object that contains the percentage and status message of the current task.
        Returns:
             None:
        """
        if status_data is None:
            return

        if 0 <= status_data.percent <= 100:
            print(f"{status_data.percent}%: {status_data.message}")
            logging.info(f"{status_data.percent}%: {status_data.message}", extra={'user': 'SGT Logs'})

        if status_data.type == "info":
            print(f"{status_data.message}")
            logging.info(f"{status_data.message}", extra={'user': 'SGT Logs'})
        elif status_data.type == "error":
            print(f"Error: {status_data.message}")
            logging.exception(f"{status_data.message}", extra={'user': 'SGT Logs'})

    @classmethod
    def start(cls) -> None:
        """Initializes and starts the terminal/CMD the StructuralGT application."""

        # Retrieve user settings
        opt_parser = OptionParser()
        opt_parser.add_option('-f', '--inputFile',
                             dest='file_path',
                             help='path to image file',
                             default="../datasets/InVitroBioFilm.png",
                             type='string')
        opt_parser.add_option('-d', '--inputDir',
                             dest='img_dir_path',
                             help='path to folder containing images',
                             default="",
                             type='string')
        opt_parser.add_option('-o', '--outputDir',
                              dest='output_dir',
                              help='path to folder for saving output files. If not provided, output files will be saved in input dir.',
                              default="",
                              type='string')
        opt_parser.add_option('-s', '--allowAutoScale',
                             dest='auto_scale',
                             help='allow automatic scaling of images',
                             default=1,
                             type='int')
        opt_parser.add_option('-t', '--runTask',
                              dest='run_task',
                              help='you can run the following tasks: (1) extract graph; (2) compute GT metrics.',
                              default=2,
                              type='int')
        opt_parser.add_option('-c', '--config',
                              dest='config_file',
                              help='path to config file',
                              default="",
                              type='string')
        # opt_parser.add_option('-m', '--runMultiGT',
        #                      dest='run_multi_gt',
        #                      help='run compute GT parameters on multiple images',
        #                      default=0,
        #                      type='int')
        # opt_parser.add_option('-i', '--selectedImgIndex',
        #                      dest='sel_img_idx',
        #                      help='index of selected image',
        #                      default=0,
        #                      type='int')
        (cfg, args) = opt_parser.parse_args()
        cfg.auto_scale = bool(cfg.auto_scale)
        # cfg.run_multi_gt = bool(cfg.run_multi_gt)

        # 1. Create Terminal App
        term_app = cls(cfg.config_file)

        # 2. Verify image files
        term_app.check_image_files(img_path=cfg.file_path, img_dir=cfg.img_dir_path, out_dir=cfg.output_dir)

        # 3. Execute specific task
        if cfg.run_task == 0:
            pass
        elif cfg.run_task == 1:
            sgt_obj = term_app.get_selected_sgt_obj()
            status, result = term_app._task_worker.task_extract_graph(sgt_obj.ntwk_p)
            TerminalApp.task_finished(status, result)
        elif cfg.run_task == 2:
            run_multi_gt = True if cfg.img_dir_path != "" else False
            if run_multi_gt:
                term_app.replicate_sgt_configs()
                status, result = term_app._task_worker.task_compute_multi_gt(term_app._sgt_objs)
                TerminalApp.task_finished(status, result)
            else:
                status, result = term_app._task_worker.task_compute_gt(term_app.get_selected_sgt_obj())
                TerminalApp.task_finished(status, result)
        else:
            term_app.update_progress(ProgressData(type="error", sender="GT", message=f"Invalid GT task selected! System will exit."))
            sys.exit('System exit')
