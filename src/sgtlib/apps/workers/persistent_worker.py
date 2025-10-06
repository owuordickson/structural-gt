# SPDX-License-Identifier: GNU GPL v3

"""
A persistent process worker for running long GT and AI-search jobs.
"""


import gc
import threading
from multiprocessing import Process, Queue
from PySide6.QtCore import QObject, Signal


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
            # result_queue.put((False, str(e)))
            print(f"Worker Loop Exception: {e}")


class ProgressListener(QObject):
    """
    Thread that listens to the multiprocessing.Queue and emits signals into QML UI.
    """
    progress = Signal(object)
    finished = Signal(bool, object)

    def __init__(self, queue: Queue, parent=None):
        super().__init__(parent)
        self._updates_queue = queue
        self._running = True
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def _listen(self):
        while self._running:
            try:
                status, payload = self._updates_queue.get()  # blocking wait
                if status == "STOP":
                    self._running = False
                    break  # Poison pill to stop the thread, otherwise keep running

                if type(status) is str:
                    self.progress.emit(payload)
                else:
                    self.finished.emit(status, payload)
            except Exception as e:
                print(f"Status Thread Listener Exception: {e}")

    def stop(self):
        try:
             self._updates_queue.put_nowait(("STOP", None))  # wakes up the blocking get()
        except Exception as e:
             print(f"Thread Listener Exception: {e}")


class PersistentProcessWorker(QObject):

    # workerStarted = Signal()
    inProgress = Signal(object)
    taskCompleted = Signal(int, bool, object)  # worker-id, success/fail, result (object)

    def __init__(self, worker_id, parent=None):
        """
        Creates a 'multiprocessing Process' to perform long-running tasks in the background without affecting the UI
        thread. This has several advantages: (1) each process is assigned its own Python interpreter and memory - this is
        an advantage for CPU-heavy jobs like image processing; (2) bypasses GIL - meaning that multiple processes can
        truly run in parallel on multiple CPU cores; (3) Processors can be forced to terminate immediately - this is not
        possible with Threads once they start; (4) if a Processor crashes, it does not affect the UI thread.

        The drawbacks are: (1) unlike with Threads, memory is not shared - objects must be serialized/pickled, (2)
        start-up cost is higher than Threads, (3) since memory is not shared, communication is via Queues/Pipes.

        We overcome the start-up cost (which causes sluggish-ness) by using Persistent Processes that are started during
        application launch. To clear and release memory, we restart the Processes after every 3 jobs are completed.

        Args:
            worker_id: The unique ID of the Persistent process worker.
            parent: The parent QObject.
        """
        super().__init__(parent)
        self._worker_id = worker_id
        self._job_queue = None
        self._status_queue = None
        self._process = None
        self._waiting = False
        self._status_listener = None
        self._task_count = 1
        self._start()

    @property
    def status_queue(self):
        return self._status_queue

    @property
    def task_count(self):
        return self._task_count

    def _start(self):
        """Start the worker process and the status listener thread."""
        if self._process is None or not self._process.is_alive():
            # start the persistent process
            self._job_queue = Queue()
            self._status_queue = Queue()
            self._process = Process(target=_worker_loop, args=(self._job_queue, self._status_queue))
            self._process.start()

            if self._status_listener:
                self._status_listener.stop()
            # start a progress/status listener thread
            self._status_listener = ProgressListener(self._status_queue)
            self._status_listener.progress.connect(self.inProgress)
            self._status_listener.finished.connect(self.on_finished)
            # self._status_listener.finished.connect(lambda success, result: self.taskFinishedSignal.emit(self._worker_id, success, result))
            # self.workerStarted.emit()

    def stop(self):
        """Force terminate the worker process."""
        if self._status_listener:
            self._status_listener.stop()
        self._status_listener = None

        if self._process and self._process.is_alive():
            # stop process
            try:
                self._job_queue.put_nowait(None)  # send poison pill
            except Exception as e:
                print(f"Unable to stop, job queue is full: {e}")
            self._process.terminate()
            self._process.join()
        self._process = None
        self._job_queue.close()
        self._status_queue.close()

    """
    # Not effective because it does not release memory - memory is only released when constructor is used to create a
    # new object.
    def restart(self):
        self.workerStarted.connect(lambda : self.on_finished(True, None))
        self.stop()
        self._start()
    """

    def on_finished(self, success, result):
        self._waiting = False
        self.taskCompleted.emit(self._worker_id, success, result)
        # Trigger GC after the job finishes
        gc.collect()
        self._task_count += 1

    def submit_task(self, func, args=()):
        if self._waiting:
            return False  # already busy
        try:
            self._job_queue.put_nowait((func, args))
            self._waiting = True
        except Exception as e:
            print(f"Job queue is full: {e}")
            return False
        return True
