import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

// Icons retrieved from Iconfinder.com and used under the CC0 1.0 Universal Public Domain Dedication.

Rectangle {
    width: parent.width
    height: 40
    color: "#f0f0f0"
    border.color: "#d0d0d0"
    //border.width: 2

    RowLayout {
        //anchors.fill: parent
        anchors.right: parent.right


        Row {
            Layout.alignment: Qt.AlignVCenter

            Button {
                id: btnCrop
                text: ""
                icon.source: "assets/icons/crop_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                ToolTip.text: "Crop image"
                ToolTip.visible: btnCrop.hovered
            }

            Button {
                id: btnBrightness
                text: ""
                icon.source: "assets/icons/brightness_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                ToolTip.text: "Adjust brightness/contrast"
                ToolTip.visible: btnBrightness.hovered
            }
        }

        Rectangle {
            width: 1
            height: 25
            color: "#d0d0d0"
            Layout.alignment: Qt.AlignVCenter
        }

        RowLayout {
            Layout.alignment: Qt.AlignVCenter

            ComboBox {
                id: cbImageType
                Layout.minimumWidth: 150
                model: ListModel {
                    ListElement { text: "Original Image"; value: 1 }
                    ListElement { text: "Binary Image"; value: 2 }
                    ListElement { text: "Processed Image"; value: 3 }
                }
                implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                textRole: "text"
                valueRole: "value"
            }

            Button {
                id: btnShowGraph
                text: ""
                icon.source: "assets/icons/graph_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                ToolTip.text: "Show graph"
                ToolTip.visible: btnShowGraph.hovered
            }
        }
    }

}
