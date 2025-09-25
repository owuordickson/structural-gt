# SPDX-License-Identifier: GNU GPL v3

"""
Process worker class for running StructuralGT tasks in the background.
"""

from multiprocessing import Process, Queue
from PySide6.QtCore import QObject, Signal, QThread


def _run_wrapper(func, args, queue):
    """Runs in the subprocess — executes func and puts result/error in queue."""
    try:
        success, data = func(*args)
        queue.put((success, data))
    except Exception as e:
        queue.put((False, str(e)))


class QueueListener(QThread):
    """Thread that listens to the multiprocessing.Queue and emits signals into QML UI."""
    progress = Signal(object)
    finished = Signal(bool, object)

    def __init__(self, queue: Queue):
        super().__init__()
        self._updates_queue = queue
        self._running = True

    def run(self):
        while self._running:
            try:
                status, payload = self._updates_queue.get()  # blocking wait
                if status == "STOP":
                    break

                if type(status) is str:
                    self.progress.emit(payload)
                else:
                    self.finished.emit(status, payload)
                    break  # stop after completion
            except Exception as e:
                self.finished.emit(False, str(e))
                break

    def stop(self):
        self._running = False
        try:
            self._updates_queue.put_nowait(("STOP", None))  # wakes up the blocking get()
        except Exception as e:
            print(f"Thread Listener Exception: {e}")
            pass


class ProcessWorker(QObject):
    """Wrapper around multiprocessing.Process for QML integration."""

    inProgressSignal = Signal(object)
    taskFinishedSignal = Signal(int, bool, object)  # worker-id, success/fail, result (object)

    def __init__(self, worker_id, func, args=(), parent=None):
        super().__init__(parent)
        self.func = func
        self.args = args
        self._worker_id = worker_id
        self._process = None
        self._queue = Queue()
        self._listener = None

    @property
    def queue(self):
        return self._queue

    def start(self):
        """Start the worker process."""
        if self._process is None or not self._process.is_alive():
            self._process = Process(target=_run_wrapper, args=(self.func, self.args, self._queue))
            self._process.start()

            # start queue listener thread
            self._listener = QueueListener(self._queue)
            self._listener.progress.connect(self.inProgressSignal)
            self._listener.finished.connect(
                lambda success, result: self.taskFinishedSignal.emit(self._worker_id, success, result)
            )
            self._listener.start()

    def stop(self):
        """Force terminate the worker process."""
        if self._process and self._process.is_alive():
            self._process.terminate()
            self._process.join()
        self._process = None

        if self._listener and self._listener.isRunning():
            self._listener.stop()
            self._listener.quit()
            self._listener.wait()
        self._listener = None
