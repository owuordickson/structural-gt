import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
//import Qt5Compat.GraphicalEffects
import "../widgets"

// Icons retrieved from Iconfinder.com and used under the CC0 1.0 Universal Public Domain Dedication.

Rectangle {
    id: rectRibbon
    width: parent.width
    height: 40
    radius: 5
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    border.width: 1

    /*DropShadow {
        anchors.fill: rectRibbon
        source: rectRibbon
        horizontalOffset: 0
        verticalOffset: 5
        radius: 1
        samples: 16
        color: "black"
        opacity: 0.5
    }

    Rectangle {
        anchors.fill: rectRibbon
        radius: 5
        color: "#f0f0f0" // the rectangle's own background
        border.color: "#d0d0d0"
        border.width: 1
    }*/

    RowLayout {
        //anchors.fill: parent
        anchors.right: parent.right

        Row {
            Layout.alignment: Qt.AlignVCenter

            Button {
                id: btnSelect
                text: ""
                icon.source: "../assets/icons/square_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: transientParent
                ToolTip.text: "Select area to crop"
                ToolTip.visible: btnSelect.hovered
            }

            Button {
                id: btnCrop
                text: ""
                icon.source: "../assets/icons/crop_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: transientParent
                ToolTip.text: "Crop to selection"
                ToolTip.visible: btnCrop.hovered
                visible: false
            }

            Button {
                id: btnBrightness
                text: ""
                icon.source: "../assets/icons/brightness_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: transientParent
                ToolTip.text: "Adjust brightness/contrast"
                ToolTip.visible: btnBrightness.hovered
                onClicked: dialogBrightnessCtrl.open()
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
                Layout.rightMargin: 10
                text: ""
                icon.source: "../assets/icons/graph_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: transientParent
                ToolTip.text: "Show graph"
                ToolTip.visible: btnShowGraph.hovered
                onClicked: dialogShowGraph.open()
            }
        }
    }

}


