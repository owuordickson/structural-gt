import QtQuick
import QtQuick.Controls
//import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window

Window {
    id: imgHistogramWindow
    width: 640
    height: 400
    x: 1024  // Exactly starts where your app ends
    y: 40
    //flags: Qt.Window | Qt.FramelessWindowHint
    visible: false  // Only show when needed
    title: "Histogram of Processed Image"



    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
        }
    }
}