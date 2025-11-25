# SPDX-License-Identifier: GNU GPL v3
"""
Checks OS-level theme
"""

from PySide6.QtCore import QObject, Property, Signal, Slot
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication


class ThemeManager(QObject):
    themeChanged = Signal()

    def __init__(self):
        super().__init__()
        self._isDark = ThemeManager.detect_dark_mode()

    @Property(bool, notify=themeChanged)
    def is_dark(self):
        return self._isDark

    @Slot()
    def reload_theme(self):
        """Call this if the OS theme changes (Windows 11 sends events)."""
        new_state = ThemeManager.detect_dark_mode()
        if new_state != self._isDark:
            self._isDark = new_state
            self.themeChanged.emit()

    # Detect OS theme
    @staticmethod
    def detect_dark_mode():
        palette = QApplication.palette()
        # simple but works on Windows/macOS/Linux
        return palette.color(QPalette.ColorRole.Window).value() < 128
