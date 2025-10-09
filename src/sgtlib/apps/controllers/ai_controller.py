# SPDX-License-Identifier: GNU GPL v3
"""
Pyside6 controller class for AI search GUI components.
"""


import logging
from PySide6.QtCore import Signal, Slot, QObject, Property

from ..models.checkbox_model import CheckBoxModel
from ...compute.graph_analyzer import GraphAnalyzer


class AIController(QObject):

    _aiBusyChanged = Signal()
    _aiModeChanged = Signal()
    updateAIProgressSignal = Signal(int, str)

    def __init__(self, controller_obj, parent: QObject = None):
        super().__init__(parent)
        self._ctrl = controller_obj
        self._ai_mode_active = False
        self._wait_flag_ai = False

        # Create Models
        self.aiSearchModel = CheckBoxModel([])
        # Attach listener for syncing models
        self._ctrl.syncModelSignal.connect(self.synchronize_ai_models)

    @Property(bool, notify=_aiBusyChanged)
    def ai_busy(self):
        return self._wait_flag_ai

    @Property(bool, notify=_aiModeChanged)
    def ai_mode_active(self):
        return self._ai_mode_active

    def start_task(self):
        """Activate the AI running (or busy) flag."""
        self._wait_flag_ai = True
        self._aiBusyChanged.emit()

    def stop_task(self):
        """Deactivate the AI running (or busy) flag."""
        self._wait_flag_ai = False
        self._aiBusyChanged.emit()

    def synchronize_ai_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload image configuration selections and controls from saved dict to QML gui_mcw after the image is loaded.

            :param sgt_obj: A GraphAnalyzer object with all saved user-selected configurations.
        """
        if sgt_obj is None:
            return

        try:
            # Models Auto-update with saved sgt_obj configs. No need to re-assign!
            ntwk_p = sgt_obj.ntwk_p
            options_ai = ntwk_p.configs

            # Get data from object configs
            ai_search_params = [v for v in options_ai.values() if v["type"] == "search-params"]

            # Update QML adapter-models with fetched data
            self.aiSearchModel.reset_data(ai_search_params)
        except Exception as err:
            logging.exception("Fatal Error: %s", err, extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Fatal Error", "Error re-loading AI configurations! Close app and try again.")

    @Slot(bool)
    def toggle_ai_mode(self, activate):
        """Toggle AI mode."""
        self._ai_mode_active = activate
        # if not activate:
        #    self._stop_ai_search()
        self._aiModeChanged.emit()

    @Slot()
    def run_ai_filter_search(self):
        """Run AI filter search on the selected SGT object."""
        if not self._ai_mode_active:
            return

        if self._wait_flag_ai:
            logging.info("Another AI task is running!", extra={'user': 'SGT Logs'})
            self._ctrl.showAlertSignal.emit("Please Wait", "Another AI task is running!")
            return

        try:
            self.start_task()
            sgt_obj = self._ctrl.get_selected_sgt_obj()
            self._ctrl.submit_job(2, "Metaheuristic-Search", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            self.stop_task()
            logging.info("AI Mode Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot()
    def reset_ai_filter_results(self):
        """Reset the results by moving the best candidate to the ignore list"""
        sgt_obj = self._ctrl.get_selected_sgt_obj()
        sgt_obj.ntwk_p.reset_metaheuristic_search()
        self.run_ai_filter_search()


