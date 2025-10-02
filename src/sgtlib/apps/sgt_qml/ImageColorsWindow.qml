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
    y: 100
    visible: false  // Only show when needed
    title: "Image Colors"

    ColumnLayout {
        anchors.fill: parent

        // Image Selection Layout
        RowLayout {
            spacing: 2
            Layout.margins: 5
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            //Layout.preferredHeight: 28
        }

        // Retrieve button and spinner -- Layout (hidden if ImageColors is visible)
        RowLayout {
            spacing: 2
            Layout.margins: 5
            Layout.fillWidth: true
            //Layout.fillHeight: true
            Layout.alignment: Qt.AlignHCenter
            visible: true

            Button {
                id: btnGetColors
                text: " Retrieve Colors"
                leftPadding: 10
                rightPadding: 10
                icon.source: "assets/icons/reload_icon.png"
                icon.width: 21
                icon.height: 21
                icon.color: "transparent"   // important for PNGs
                ToolTip.text: "Get the dominant colors of the image."
                ToolTip.visible: btnGetColors.hovered
                //visible: !mainController.processing_colors
                //onClicked: mainController.()
            }

            Column {
                //visible: mainController.processing_colors

                SpinnerProgress {
                    //running: mainController.processing_colors
                    width: 24
                    height: 24
                }
            }
        }

        // Image Colors Layout (hidden if RetrieveButton is visible)
        RowLayout {
            spacing: 2
            Layout.margins: 5
            Layout.fillWidth: true
            //Layout.fillHeight: true
            Layout.alignment: Qt.AlignHCenter
            visible: false
        }

    }


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