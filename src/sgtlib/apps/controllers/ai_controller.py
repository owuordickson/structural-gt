# SPDX-License-Identifier: GNU GPL v3
"""
Pyside6 controller class for AI search GUI components.
"""

from PySide6.QtCore import Signal, Slot, QObject

from ..models.checkbox_model import CheckBoxModel


class AIController(QObject):

    updateAIProgressSignal = Signal(int, str)
    _aiBusyChanged = Signal()
    _aiModeChanged = Signal()

    def __init__(self, parent: QObject = None):
        super().__init__(parent)

        # Create Models
        self.aiSearchModel = CheckBoxModel([])

    def synchronize_ai_models(self, sgt_obj: GraphAnalyzer):
        """
            Reload image configuration selections and controls from saved dict to QML gui_mcw after the image is loaded.

            :param sgt_obj: A GraphAnalyzer object with all saved user-selected configurations.
        """
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
            self.showAlertSignal.emit("Fatal Error", "Error re-loading AI configurations! Close app and try again.")

    @Slot()
    def run_ai_filter_search(self):
        """Run AI filter search on the selected SGT object."""
        if not self._ai_mode_active:
            return

        if self._wait_flag_ai:
            logging.info("Another AI task is running!", extra={'user': 'SGT Logs'})
            self.showAlertSignal.emit("Please Wait", "Another AI task is running!")
            return

        try:
            self._start_ai_task()
            sgt_obj = self.get_selected_sgt_obj()
            self._submit_job(2, "Metaheuristic-Search", (sgt_obj.ntwk_p,), True)
        except Exception as err:
            self._stop_ai_task()
            logging.info("AI Mode Error: %s", err, extra={'user': 'SGT Logs'})

    @Slot()
    def reset_ai_filter_results(self):
        """Reset the results by moving the best candidate to the ignore list"""
        sgt_obj = self.get_selected_sgt_obj()
        sgt_obj.ntwk_p.reset_metaheuristic_search()
        self.run_ai_filter_search()


