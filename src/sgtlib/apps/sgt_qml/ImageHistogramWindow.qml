import QtQuick
import QtQuick.Controls
//import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window
import "widgets"

Window {
    id: imgHistogramWindow
    width: 768
    height: 720
    x: 1024  // Exactly starts where your app ends
    y: 40
    //flags: Qt.Window | Qt.FramelessWindowHint
    visible: false  // Only show when needed
    title: "Histogram of Processed Image(s)"

    ColumnLayout {
        anchors.fill: parent

        // Reload button and spinner -- Layout
        RowLayout {
            spacing: 2
            Layout.margins: 5
            Layout.alignment: Qt.AlignHCenter

            Button {
                id: btnReloadHistogram
                text: " Reload Histogram"
                leftPadding: 10
                rightPadding: 10
                icon.source: "assets/icons/reload_icon.png"
                icon.width: 21
                icon.height: 21
                icon.color: "transparent"   // important for PNGs
                ToolTip.text: "Reload everytime you change the image."
                ToolTip.visible: btnReloadHistogram.hovered
                visible: !mainController.histogram_busy
                onClicked: mainController.compute_img_histogram()
            }

            Column {
                visible: mainController.histogram_busy

                SpinnerProgress {
                    running: mainController.histogram_busy
                    width: 24
                    height: 24
                }
            }
        }

        // Histogram grid view -- Layout
        ScrollView {
            Layout.fillWidth: true
            Layout.preferredHeight: imgHistogramWindow.height - 10
            clip: true  // Ensures contents are clipped to the scroll view bounds

            GridView {
                id: imgHistGridView
                anchors.fill: parent
                cellWidth: (parent.width / 2)
                cellHeight: (parent.height / 2)
                model: imgHistogramModel
                visible: true

                delegate: Item {
                    width: imgHistGridView.cellWidth
                    height: imgHistGridView.cellHeight

                    Rectangle {
                        width: parent.width - 2  // Adds horizontal spacing
                        height: parent.height - 2  // Adds vertical spacing
                        color: "#ffffff"  // Background color for spacing effect
                        visible: model.selected === 1

                        Image {
                            source: model.image === "" ? "" : "data:image/png;base64," + model.image  // Base64 encoded image
                            width: parent.width
                            height: parent.height
                            anchors.centerIn: parent
                            transformOrigin: Item.Center
                            fillMode: Image.PreserveAspectFit
                        }

                        Label {
                            text: "Frame " + model.id
                            color: "#bc0022"
                            anchors.left: parent.left
                            anchors.top: parent.top
                            anchors.margins: 2
                            background: Rectangle {
                                color: "transparent"
                            }
                        }

                    }

                }

            }

        }
    }


    Connections {
        target: mainController

        function onShowImageHistogramSignal(allow) {
            // Force refresh
            if (imgHistogramWindow.visible) {
                imgHistogramWindow.visible = allow;
            }
        }
    }
}