# SPDX-License-Identifier: GNU GPL v3
# This file is dual licensed under the terms of the GNU General Public, Version
# 3.0.  See the LICENSE file in the root of this
# repository for complete details.

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
        >>>     print(f"{code}: {msg}")
        >>>
        >>> upd = ProgressUpdate()
        >>> upd.add_listener(print_progress)  # to get updates
        >>> upd.update_status([1,"Sending update ..."])
        >>> upd.remove_listener(print_progress) # to opt out of updates
        >>>
        """
        self.__listeners = []
        self.abort = False

    def abort_tasks(self):
        """
        Set abort flag.
        :return:
        """
        self.abort = True

    def add_listener(self, func: ()):
        """
        Add functions from the list of listeners.
        :param func:
        :return:
        """
        if func in self.__listeners:
            return
        self.__listeners.append(func)

    def remove_listener(self, func):
        """
        Remove functions from the list of listeners.
        :param func:
        :return:
        """
        if func not in self.__listeners:
            return
        self.__listeners.remove(func)

    def update_status(self, args=None):
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
