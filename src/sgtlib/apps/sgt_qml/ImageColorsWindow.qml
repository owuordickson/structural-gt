import QtQuick
import QtQuick.Controls
//import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window
import "widgets"

Window {
    id: imgColorsWindow
    width: 768
    height: 720
    x: 1024  // Exactly starts where your app ends
    y: 40
    visible: false  // Only show when needed
    title: "Image Colors"



    Connections {
        target: mainController

        function onShowImageHistogramSignal(allow) {
            // Force refresh
            if (imgColorsWindow.visible) {
                imgColorsWindow.visible = allow;
            }
        }
    }

}