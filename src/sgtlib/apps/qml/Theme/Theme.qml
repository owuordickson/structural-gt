pragma Singleton
import QtQuick

QtObject {
    readonly property color black:       themeManager.is_dark ? "#ffffff" : "#000000"
    readonly property color gray:       themeManager.is_dark ? "#909090" : "#909090"
    readonly property color veryLightGray: themeManager.is_dark ? "#555" : "#e5e5e5"
    readonly property color lightGray:     themeManager.is_dark ? "#333" : "#d0d0d0"  // border color
    readonly property color darkGray:       themeManager.is_dark ? "#606060" : "#303030"
    readonly property color green:    themeManager.is_dark ? "#22bc55" : "#22bc55"
    readonly property color lightGreen:     themeManager.is_dark ? "#f0fff0" : "#f0fff0"
    readonly property color darkGreen:       themeManager.is_dark ? "#008b00" : "#008b00"
    readonly property color white:       themeManager.is_dark ? "#000000" : "#ffffff"
    readonly property color smokeWhite:       themeManager.is_dark ? "#f5f5f5" : "#f5f5f5"
    readonly property color blue:       themeManager.is_dark ? "#2266ff" : "#2266ff"
    readonly property color skyBlue:       themeManager.is_dark ? "#00aeef" : "#00aeef"
    readonly property color dodgerBlue:       themeManager.is_dark ? "#2299ff" : "#2299ff"
    readonly property color yellow:       themeManager.is_dark ? "#ffde21" : "#ffde21"
    readonly property color red:    themeManager.is_dark ? "#bc0000" : "#bc0000"

    readonly property color text:       themeManager.is_dark ? "#f0f0f0" : "#1a1a1a"
    readonly property color background: themeManager.is_dark ? "#000000" : "#f0f0f0"
    readonly property color semiTransparent:     themeManager.is_dark ? "#80ffffff" : "#50000000" // 80% or 50% opacity
    //readonly property color disabled:   themeManager.is_dark ? "#777" : "#999"
}