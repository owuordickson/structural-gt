# SPDX-License-Identifier: GNU GPL v3

"""
Process worker class for running StructuralGT tasks in the background.
"""

from multiprocessing import Process, Queue
from PySide6.QtCore import QObject, Signal


def _run_wrapper(func, args, queue):
    """Runs in the subprocess — executes func and puts result/error in queue."""
    try:
        success, data = func(*args)
        queue.put((success, data))
    except Exception as e:
        queue.put((False, str(e)))


class ProcessWorker(QObject):
    """Wrapper around multiprocessing.Process for QML integration."""

    inProgressSignal = Signal(int, str)  # progress-value (0-100), progress-message (str)
    taskFinishedSignal = Signal(bool, object)  # success/fail (True/False), result (object)

    def __init__(self, func, args=(), parent=None):
        super().__init__(parent)
        self.func = func
        self.args = args
        self._process = None
        self._queue = Queue()

    @property
    def queue(self):
        return self._queue

    def start(self):
        """Start the worker process."""
        if self._process is None or not self._process.is_alive():
            self._process = Process(target=_run_wrapper, args=(self.func, self.args, self._queue))
            self._process.start()

    def stop(self):
        """Force terminate the worker process."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._process = None

    def poll(self):
        """Check if the worker finished and emit signals (should be polled by a QTimer)."""
        try:
            while not self._queue.empty():
                status, payload = self._queue.get_nowait()
                if type(status) is str:
                    percent, message = payload
                    self.inProgressSignal.emit(int(percent), message)
                else:
                    self.taskFinishedSignal.emit(status, payload)
        except Exception as e:
            self.taskFinishedSignal.emit(False, f"Error: {e}")