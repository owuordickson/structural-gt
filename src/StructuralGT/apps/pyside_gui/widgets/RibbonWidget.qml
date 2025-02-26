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

        RowLayout {
            Layout.alignment: Qt.AlignVCenter

            Button {
                id: btnSelect
                text: ""
                Layout.fillHeight: true
                Layout.fillWidth: true
                background: transientParent  // bgcolor same as parent color (i.e., rectRibbon)
                ToolTip.text: "Select area to crop"
                ToolTip.visible: btnSelect.hovered
                visible: mainController.display_image();
                onClicked: enableRectangularSelect()

                Rectangle {
                    id: btnSelectBorder
                    width: 20
                    height: 20
                    radius: 4
                    color: "transparent"
                    border.width: 2
                    border.color: "black"
                    anchors.centerIn: parent
                    //enabled: false
                }
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
                onClicked: mainController.perform_cropping(true)
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
                enabled: mainController.display_image();
            }

            Button {
                id: btnUndo
                text: ""
                icon.source: "../assets/icons/undo_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: transientParent
                ToolTip.text: "Undo crop"
                ToolTip.visible: btnUndo.hovered
                onClicked: mainController.undo_cropping(true)
                visible: false
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
                    id: imgTypeModel
                    ListElement { text: "Original Image"; value: 0 }
                    ListElement { text: "Binary Image"; value: 3 }
                    ListElement { text: "Processed Image"; value: 2 }
                    ListElement { text: "Extracted Graph"; value: 4 }
                }
                implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                textRole: "text"
                valueRole: "value"
                enabled: mainController.display_image();
                onCurrentIndexChanged: mainController.select_img_type(valueAt(currentIndex))
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
                onClicked: dialogExtractGraph.open()
                enabled: mainController.display_image();
            }
        }
    }

    function enableRectangularSelect() {
        if (btnSelectBorder.enabled) {
            mainController.enable_rectangular_selection(false)
            btnSelectBorder.border.color = "black"
            btnSelectBorder.enabled = false
        } else {
            mainController.enable_rectangular_selection(true)
            btnSelectBorder.border.color = "red"
            btnSelectBorder.enabled = true
        }
    }

    Connections {
        target: mainController

        function onShowCroppingToolSignal(allow) {
            if (allow) {
                btnCrop.visible = true;
            } else {
                btnCrop.visible = false
            }
        }

        function onShowUnCroppingToolSignal(allow) {
            if (allow) {
                btnUndo.visible = true
            } else {
                btnUndo.visible = false
            }
        }

        function onImageChangedSignal() {
            // Force refresh
            btnSelect.visible = mainController.display_image();
            btnBrightness.enabled = mainController.display_image();
            cbImageType.enabled = mainController.display_image();
            btnShowGraph.enabled = mainController.display_image();

            let curr_type = mainController.get_current_img_type();
            for (let i=0; i < cbImageType.model.count; i++) {
                if (cbImageType.model.get(i).value === curr_type){
                    cbImageType.currentIndex = i;
                }
            }
        }
    }

}


