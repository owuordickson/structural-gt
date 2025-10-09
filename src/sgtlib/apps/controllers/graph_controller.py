# SPDX-License-Identifier: GNU GPL v3
"""
Pyside6 (GUI components) controller class for graph extraction and computation.
"""

import logging
from PySide6.QtCore import Slot, QObject

from ..models.tree_model import TreeModel
from ..models.table_model import TableModel
from ..models.checkbox_model import CheckBoxModel
from ...utils.sgt_utils import ProgressData
from ...compute.graph_analyzer import GraphAnalyzer


class GraphController(QObject):

    def __init__(self, controller_obj, parent: QObject = None):
        super().__init__(parent)
        self._ctrl = controller_obj

        # Create Models
        self.graphPropsModel = TableModel([])
        self.graphComputeModel = TableModel([])
        self.gtcScalingModel = CheckBoxModel([])
        self.gteTreeModel = TreeModel([])
        self.gtcListModel = CheckBoxModel([])
        self.exportGraphModel = CheckBoxModel([])

        # Attach listener for syncing models
        self._ctrl.syncModelSignal.connect(self.synchronize_graph_models)

    def start_task(self, msg: str = "please wait..."):
        """Activate the wait flag and send a wait signal."""
        self._ctrl.wait_msg = msg
        self._ctrl.wait_flag = True

    def stop_task(self):
        """Deactivate the wait flag and send a wait signal."""
        self._ctrl.wait_msg = ""
        self._ctrl.wait_flag = False

    def synchronize_graph_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload graph configuration selections and controls from saved dict to QML gui_mcw.
        Args:
            sgt_obj: a GraphAnalyzer object with all saved user-selected configurations.

        Returns:

        """
        if sgt_obj is None:
            return

        try:
            # Models Auto-update with saved sgt_obj configs. No need to re-assign!
            ntwk_p = sgt_obj.ntwk_p
            graph_obj = ntwk_p.graph_obj
            option_gte = graph_obj.configs
            options_gtc = sgt_obj.configs

            graph_options = [v for v in option_gte.values() if v["type"] == "graph-extraction"]
            file_options = [v for v in option_gte.values() if v["type"] == "file-options"]
            compute_options = [v for v in options_gtc.values() if v["type"] == "gt-metric"]
            scaling_options = [v for v in options_gtc.values() if v["type"] == "scaling-param"]

            self.gteTreeModel.reset_data(graph_options)
            self.exportGraphModel.reset_data(file_options)
            self.gtcListModel.reset_data(compute_options)
            self.gtcScalingModel.reset_data(scaling_options)

            self.graphPropsModel.reset_data(graph_obj.props)
            self.graphComputeModel.reset_data(sgt_obj.props)
        except Exception as err:
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Fatal Error", "Error re-loading image configurations! Close app and try again.")

    @Slot(result=bool)
    def display_graph(self):
        if len(self._ctrl.sgt_objs) <= 0:
            return False

        sgt_obj = self._ctrl.get_selected_sgt_obj()
        if sgt_obj is None:
            return False

        if sgt_obj.ntwk_p.graph_obj.img_ntwk is None:
            return False

        if sgt_obj.ntwk_p.selected_batch_view == "graph":
            return True
        return False

    @Slot(bool)
    def reload_graph_image(self, only_giant_graph=False):
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another GT task is running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another GT task is running!")
            return

        try:
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            sel_img_batch = sgt_obj.ntwk_p.selected_batch
            sgt_obj.ntwk_p.draw_graph_image(sel_img_batch, show_giant_only=only_giant_graph)
            self._ctrl.changeImageSignal.emit()
            self.stop_task()
        except Exception as err:
            self.stop_task()
            logging.exception("Error reloading graph image: %s", err, extra={'user': 'SGT Logs'})

    @Slot()
    def load_graph_simulation(self):
        """Render and visualize OVITO graph network simulation."""
        try:
            # Import libraries
            from ovito import scene
            from ovito.vis import Viewport
            from ovito.io import import_file
            from ovito.gui import create_qwidget

            # Clear any existing scene
            for p_line in list(scene.pipelines):
                p_line.remove_from_scene()

            # Create OVITO data pipeline
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            h, w = sgt_obj.ntwk_p.graph_obj.img_ntwk.shape[:2]
            pipeline = import_file(sgt_obj.ntwk_p.graph_obj.gsd_file)
            pipeline.add_to_scene()

            vp = Viewport(type=Viewport.Type.Perspective, camera_dir=(2, 1, -1))
            vp.zoom_all((w, h))  # width, height

            ovito_widget = create_qwidget(vp, parent=self._ctrl.qml_app.activeWindow())
            ovito_widget.setFixedSize(w, h)
            ovito_widget.show()
        except Exception as e:
            logging.exception(f"Graph Simulation Error: {e}", extra={'user': 'SGT Logs'})

    @Slot()
    def export_graph_to_file(self):
        """Export graph data and save as a file."""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another GT task is running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another GT task is running!")
            return
        self._ctrl.handle_progress_update(ProgressData(percent=0, sender="GT", message=f"Exporting Graph Data..."))
        try:
            if self._ctrl.get_selected_sgt_obj().ntwk_p.selected_batch.is_graph_only:
                return

            self._ctrl.handle_progress_update(ProgressData(percent=20, sender="GT", message=f"Exporting Graph Data..."))
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(1, "Export-Graph", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            logging.exception("Unable to Export Graph: " + str(err), extra={'user': 'SGT Logs'})
            self._ctrl.handle_finished(1, False,
                                           ["Unable to Export Graph", "Error exporting graph to file. Try again."])

    @Slot()
    def run_extract_graph(self):
        """Retrieve settings from the model and send to Python."""

        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another GT task is running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another GT task is running!")
            return

        try:
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(1, "Extract-Graph", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            self.stop_task()
            logging.exception("Graph Extraction Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.handle_progress_update(
                ProgressData(type="error", sender="GT", message=f"Fatal error occurred! Close the app and try again."))
            self._ctrl.handle_finished(1, False, ["Graph Extraction Error",
                                             "Fatal error while trying to extract graph. "
                                             "Close the app and try again."])

    @Slot(float)
    def rate_graph(self, rating: float):
        """Rate extracted graph on a scale of 1-10"""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another GT task running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another GT task is running!")
            return

        try:
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(1, "Rate-Graph", (rating, sgt_obj.ntwk_p,), True)
        except Exception as err:
            self.stop_task()
            logging.info("Rate Graph Error: " + str(err), extra={'user': 'SGT Logs'})

    @Slot()
    def run_graph_analyzer(self):
        """Retrieve settings from the model and send to Python."""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another GT task is running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another GT task is running!")
            return

        try:
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(1, "Compute-GT", (sgt_obj,), True)
        except Exception as err:
            self.stop_task()
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.handle_progress_update(
                ProgressData(type="error", sender="GT", message=f"Fatal error occurred! Close the app and try again."))
            self._ctrl.handle_finished(1, False, ["GT Computation Error",
                                             "Fatal error while trying calculate GT parameters. "
                                             "Close the app and try again."])

    @Slot()
    def run_multi_graph_analyzer(self):
        """"""
        if self._ctrl.wait_flag:
            logging.info("Please Wait: Another GT task is running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another GT task is running!")
            return

        try:
            self.start_task()
            # Update Configs
            self._ctrl.replicate_sgt_configs()
            # Start Background Process
            self._ctrl.submit_job(1, "Compute-Multi-GT", (self._ctrl.sgt_objs,), True)
        except Exception as err:
            self.stop_task()
            logging.exception("GT Computation Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.handle_progress_update(
                ProgressData(type="error", sender="GT", message=f"Fatal error occurred! Close the app and try again."))
            self._ctrl.handle_finished(1, False, ["GT Computation Error",
                                             "Fatal error while trying calculate GT parameters. "
                                             "Close the app and try again."])

