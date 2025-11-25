pragma Singleton
import QtQuick

QtObject {
    readonly property color background: themeManager.is_dark ? "#000000" : "#f0f0f0"
    readonly property color grayBg: themeManager.is_dark ? "#000000" : "#e5e5e5"
    readonly property color lightGreenBg:     themeManager.is_dark ? "#f0fff0" : "#f0fff0"
    readonly property color text:       themeManager.is_dark ? "#f0f0f0" : "#1a1a1a"
    readonly property color black:       themeManager.is_dark ? "#000000" : "#000000"
    readonly property color darkGray:       themeManager.is_dark ? "#606060" : "#303030"
    readonly property color darkGreen:       themeManager.is_dark ? "#008b00" : "#008b00"
    readonly property color whiteText:       themeManager.is_dark ? "#000000" : "#ffffff"
    readonly property color smokeWhite:       themeManager.is_dark ? "#f5f5f5" : "#f5f5f5"
    readonly property color blueText:       themeManager.is_dark ? "#2266ff" : "#2266ff"
    readonly property color skyBlue:       themeManager.is_dark ? "#00aeef" : "#00aeef"
    readonly property color grayText:       themeManager.is_dark ? "#909090" : "#909090"
    readonly property color yellow:       themeManager.is_dark ? "#ffde21" : "#ffde21"
    readonly property color waitText:       themeManager.is_dark ? "#2299ff" : "#2299ff"
    readonly property color errorColor:    themeManager.is_dark ? "#bc0000" : "#bc0000"
    readonly property color successColor:    themeManager.is_dark ? "#22bc55" : "#22bc55"

    readonly property color borderColor:     themeManager.is_dark ? "#333" : "#d0d0d0"
    readonly property color tableBorderColor:     themeManager.is_dark ? "#e0e0e0" : "#e0e0e0"
    readonly property color semiTransparent:     themeManager.is_dark ? "#80ffffff" : "#50000000" // 80% or 50% opacity
    //readonly property color disabled:   themeManager.is_dark ? "#777" : "#999"
}