import QtQuick
import QtQuick.Controls
//import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window

Window {
    id: imgHistogramWindow
    width: 860
    height: 400
    x: 1024  // Exactly starts where your app ends
    y: 40
    //flags: Qt.Window | Qt.FramelessWindowHint
    visible: false  // Only show when needed
    title: "Histogram of Processed Image(s)"

    ColumnLayout {
        anchors.fill: parent

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true
            clip: true  // Ensures contents are clipped to the scroll view bounds

            GridView {
                id: imgHistGridView
                width: parent.width
                height: parent.height
                anchors.centerIn: parent
                cellWidth: parent.width / 4
                cellHeight: parent.height / 4
                model: imgHistogramModel
                visible: true

                delegate: Item {
                    width: imgHistGridView.cellWidth
                    height: imgHistGridView.cellHeight

                    Rectangle {
                        width: parent.width - 2  // Adds horizontal spacing
                        height: parent.height - 2  // Adds vertical spacing
                        color: "#d0d0d0"  // Background color for spacing effect

                        Image {
                            source: model.image === "" ? "" : "data:image/png;base64," + model.image  // Base64 encoded image
                            width: parent.width
                            height: parent.height
                            anchors.centerIn: parent
                            transformOrigin: Item.Center
                            fillMode: Image.PreserveAspectCrop
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

        function onImageChangedSignal() {
            // Force refresh
        }
    }
}