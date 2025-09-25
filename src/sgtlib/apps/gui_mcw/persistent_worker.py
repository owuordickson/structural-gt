# SPDX-License-Identifier: GNU GPL v3

"""
A persistent process worker for running long GT and AI-search jobs.
"""


from multiprocessing import Process, Queue
from PySide6.QtCore import QObject, Signal, QThread


def _worker_loop(job_queue, result_queue):
    """Persistent worker loop inside the subprocess."""
    while True:
        job = job_queue.get()
        if job is None:  # poison pill for shutdown
            break
        func, args = job

        try:
            # --- Attach result_queue to the object that will call _update_progress ---
            # If func is a bound method, its instance is func.__self__
            owner = getattr(func, "__self__", None)
            if owner is not None and hasattr(owner, "progress_queue"):
                # attach the subprocess's result_queue to the unpickled instance
                owner.attach_progress_queue(result_queue)

        # --- Run the job ---
            success, data = func(*args)
            result_queue.put((success, data))
        except Exception as e:
            result_queue.put((False, str(e)))



class ProgressListener(QThread):
    """
    Thread that listens to the multiprocessing.Queue and emits signals into QML UI.
    """
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
                    break  # Poison pill to stop the thread, otherwise keep running

                if type(status) is str:
                    self.progress.emit(payload)
                else:
                    self.finished.emit(status, payload)
            except Exception as e:
                self.finished.emit(False, str(e))

    def stop(self):
        self._running = False
        try:
            self._updates_queue.put_nowait(("STOP", None))  # wakes up the blocking get()
        except Exception as e:
            print(f"Thread Listener Exception: {e}")
            pass


class PersistentProcessWorker(QObject):

    startedSignal = Signal()
    inProgressSignal = Signal(object)
    taskFinishedSignal = Signal(int, bool, object)  # worker-id, success/fail, result (object)

    def __init__(self, worker_id, parent=None):
        super().__init__(parent)
        self._worker_id = worker_id
        self._job_queue = None
        self._status_queue = None
        self._process = None
        self._waiting = False
        self._status_listener = None
        self.start()

    @property
    def status_queue(self):
        return self._status_queue

    @property
    def status_listener(self):
        return self._status_listener

    def start(self):
        """Start the worker process and the status listener thread."""
        if self._process is None or not self._process.is_alive():
            # start the persistent process
            self._job_queue = Queue()
            self._status_queue = Queue()
            self._process = Process(target=_worker_loop, args=(self._job_queue, self._status_queue))
            self._process.start()

            # start a progress/status listener thread
            self._status_listener = ProgressListener(self._status_queue)
            self._status_listener.progress.connect(self.inProgressSignal)
            self._status_listener.finished.connect(self.on_finished)
            # self._status_listener.finished.connect(lambda success, result: self.taskFinishedSignal.emit(self._worker_id, success, result))
            self._status_listener.start()
            self.startedSignal.emit()

    def stop(self):
        """Force terminate the worker process."""
        if self._process and self._process.is_alive():
            # stop process
            self._job_queue.put(None)  # send poison pill
            self._process.terminate()
            self._process.join()
        self._job_queue = None
        self._status_queue = None
        self._process = None

        # stop progress listener thread
        if self._status_listener and self._status_listener.isRunning():
            self._status_listener.stop()
            self._status_listener.quit()
            self._status_listener.wait()
        self._status_listener = None

    def restart(self):
        """Restart the worker process."""
        self.startedSignal.connect(lambda : self.on_finished(True, None))
        self.stop()
        self.start()
        # self.on_finished(True, None)

    def on_finished(self, success, result):
        self._waiting = False
        self.taskFinishedSignal.emit(self._worker_id, success, result)

    def submit_task(self, func, args=()):
        if self._waiting:
            return False  # already busy
        self._waiting = True
        self._job_queue.put((func, args))
        return True

