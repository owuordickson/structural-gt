# SPDX-License-Identifier: GNU GPL v3

"""
Uses listener functions to send updates to outside functions.
"""


class ProgressUpdate:
    """
    A class for sending updates to outside functions.

    """
    def __init__(self):
        """
        A class for sending updates to outside functions.

        >>> def print_progress(code, msg):
        >>>     print(str(code) + ': ' + str(msg))
        >>>
        >>> upd = ProgressUpdate()
        >>> upd.add_listener(print_progress)  # to get updates
        >>> upd.update_status([1,"Sending update ..."])
        >>> upd.remove_listener(print_progress) # to opt out of updates
        >>>
        """
        self.__listeners = []
        self.abort = False

    def abort_tasks(self) -> None:
        """
        Set abort flag.
        :return:
        """
        self.abort = True

    def add_listener(self, func) -> None:
        """
        Add functions from the list of listeners.
        :param func:
        :return:
        """
        if func in self.__listeners:
            return
        self.__listeners.append(func)

    def remove_listener(self, func) -> None:
        """
        Remove functions from the list of listeners.
        :param func:
        :return:
        """
        if func not in self.__listeners:
            return
        self.__listeners.remove(func)

    def update_status(self, args=None) -> None:
        """
        Run all the functions that are saved as listeners.

        :param args:
        :return:
        """
        # Trigger events.
        if args is None:
            args = []
        for func in self.__listeners:
            func(*args)
