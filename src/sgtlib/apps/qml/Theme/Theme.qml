pragma Singleton
import QtQuick

QtObject {
    readonly property color background: themeManager.is_dark ? "#000000" : "#f0f0f0"
    readonly property color aiBackground:     themeManager.is_dark ? "#f0fff0" : "#f0fff0"
    readonly property color text:       themeManager.is_dark ? "#f0f0f0" : "#1a1a1a"
    readonly property color whiteText:       themeManager.is_dark ? "#000000" : "#ffffff"
    readonly property color blueText:       themeManager.is_dark ? "#2266ff" : "#2266ff"
    readonly property color labelText:       themeManager.is_dark ? "#909090" : "#909090"
    readonly property color waitText:       themeManager.is_dark ? "#2299ff" : "#2299ff"
    readonly property color errorColor:    themeManager.is_dark ? "#bc0000" : "#bc0000"
    readonly property color successColor:    themeManager.is_dark ? "#22bc55" : "#22bc55"

    readonly property color borderColor:     themeManager.is_dark ? "#333" : "#d0d0d0"
    //readonly property color disabled:   themeManager.is_dark ? "#777" : "#999"
}