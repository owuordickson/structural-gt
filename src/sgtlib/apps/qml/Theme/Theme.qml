pragma Singleton
import QtQuick

QtObject {
    readonly property color black:       themeManager.is_dark ? "#ffffff" : "#000000"
    readonly property color white:       themeManager.is_dark ? "#000000" : "#ffffff"
    readonly property color gray:       themeManager.is_dark ? "#c0c0c0" : "#909090"  // Label color
    readonly property color veryLightGray: themeManager.is_dark ? "#333" : "#e5e5e5"  // navigation background color
    readonly property color lightGray:     themeManager.is_dark ? "#555" : "#d0d0d0"  // border color
    readonly property color darkGray:       themeManager.is_dark ? "#e0e0e0" : "#303030"  // text color
    readonly property color darkGrey:       themeManager.is_dark ? "#606060" : "#303030"
    readonly property color green:    themeManager.is_dark ? "#22ff55" : "#22bc55"    // success color
    readonly property color smokeWhite:       themeManager.is_dark ? "#0f0f0f" : "#f5f5f5"  // Table rows color
    readonly property color blue:       themeManager.is_dark ? "#22b6ff" : "#2266ff"  // status text color
    readonly property color dodgerBlue:       themeManager.is_dark ? "#00aeef" : "#2299ff"  // progress bar color
    readonly property color red:    themeManager.is_dark ? "#ff5500" : "#bc0000"   // error color
    readonly property color yellow: "#ffdd22"
    readonly property color lightGreen: "#f0fff0"
    readonly property color darkGreen: "#008b00"

    readonly property color text:       themeManager.is_dark ? "#f0f0f0" : "#1a1a1a"
    readonly property color background: themeManager.is_dark ? "#000000" : "#f0f0f0"
    readonly property color disabled: themeManager.is_dark? "#777" : "#999"
    readonly property color semiTransparentLt: "#80ffffff" // 80% or 50% opacity
    readonly property color semiTransparentDk: "#80000000" // 80% opacity
}