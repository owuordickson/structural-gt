pragma Singleton
import QtQuick 2.15

QtObject {
    readonly property color background: themeManager.isDark ? "#000000" : "#f0f0f0"
    readonly property color text:       themeManager.isDark ? "#f0f0f0" : "#1a1a1a"
    readonly property color whiteText:       themeManager.isDark ? "#000000" : "#ffffff"
    readonly property color blueText:       themeManager.isDark ? "#2266ff" : "#2266ff"
    readonly property color labelText:       themeManager.isDark ? "#909090" : "#909090"
    readonly property color waitText:       themeManager.isDark ? "#2299ff" : "#2299ff"
    readonly property color errorColor:    themeManager.isDark ? "#bc0000" : "#bc0000"
    readonly property color successColor:    themeManager.isDark ? "#22bc55" : "#22bc55"
    //readonly property color accent:     "#3fa9f5"

    readonly property color border:     themeManager.isDark ? "#333" : "#d0d0d0"
    //readonly property color disabled:   themeManager.isDark ? "#777" : "#999"
}