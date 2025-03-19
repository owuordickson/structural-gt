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
        anchors.left: parent.left
        anchors.verticalCenter: parent.verticalCenter

        RowLayout {
            Layout.leftMargin: 5

            Button {
                id: btnHideLeftPane
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                text: ""
                property bool hidePane: true
                icon.source: hidePane ? "../assets/icons/hide_panel.png" : "../assets/icons/show_panel.png"
                icon.width: 28
                icon.height: 28
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: hidePane ? "Hide left pane" : "Show left pane"
                ToolTip.visible: btnHideLeftPane.hovered
                visible: true
                onClicked: {
                    if (hidePane) {
                        hidePane = false;
                    } else {
                        hidePane = true;
                    }
                    toggleLeftPane(hidePane);
                }
            }

        }
    }

    RowLayout {
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter

        RowLayout {

            Button {
                id: btnRescale
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                text: ""
                icon.source: "../assets/icons/rescale_icon.png" // Path to your icon
                icon.width: 20 // Adjust as needed
                icon.height: 20
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Re-scale large images"
                ToolTip.visible: btnRescale.hovered
                enabled: true //mainController.display_image()
                onClicked: drpDownRescale.open()


                Popup {
                    id: drpDownRescale
                    width: 64
                    //height: colRadioButtons.implicitHeight + 10
                    height: 100
                    modal: false
                    focus: true
                    x: 2
                    y: 32
                    background: Rectangle {
                        color: "#f0f0f0"
                        border.color: "#d0d0d0"
                        border.width: 1
                        radius: 2
                    }
                }
            }

            Button {
                id: btnBrightness
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                text: ""
                icon.source: "../assets/icons/brightness_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: Rectangle { color: "transparent"}
                ToolTip.text: "Adjust brightness/contrast"
                ToolTip.visible: btnBrightness.hovered
                onClicked: drpDownBrightness.open()
                enabled: mainController.display_image()

                Popup {
                    id: drpDownBrightness
                    width: 250
                    height: 150
                    modal: false
                    focus: true
                    x: 2
                    y: 32
                    background: Rectangle {
                        color: "#f0f0f0"
                        border.color: "#d0d0d0"
                        border.width: 1
                        radius: 2
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        BrightnessControlWidget{ id: brightnessControl }

                        RowLayout {
                            spacing: 10
                            Layout.alignment: Qt.AlignHCenter

                            Button {
                                Layout.preferredWidth: 54
                                Layout.preferredHeight: 30
                                text: ""
                                onClicked: drpDownBrightness.close()

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: "#bc0000"

                                    Label {
                                        text: "Cancel"
                                        color: "#ffffff"
                                        anchors.centerIn: parent
                                    }
                                }
                            }

                            Button {
                                Layout.preferredWidth: 40
                                Layout.preferredHeight: 30
                                text: ""
                                onClicked: drpDownBrightness.close()

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: "#22bc55"

                                    Label {
                                        text: "OK"
                                        color: "#ffffff"
                                        anchors.centerIn: parent
                                    }
                                }
                            }
                        }

                    }

                }

            }

            Button {
                id: btnSelect
                text: ""
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                background: Rectangle { color: "transparent"}
                ToolTip.text: "Select area to crop"
                ToolTip.visible: btnSelect.hovered
                visible: mainController.display_image()
                onClicked: enableRectangularSelect()

                Rectangle {
                    id: btnSelectBorder
                    width: 20
                    height: 20
                    //width: parent.width
                    //height: parent.height
                    anchors.centerIn: parent
                    radius: 4
                    color: "transparent"
                    border.width: 2
                    border.color: "black"
                }
            }

            Button {
                id: btnCrop
                text: ""
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                icon.source: "../assets/icons/crop_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: Rectangle { color: "transparent"}
                ToolTip.text: "Crop to selection"
                ToolTip.visible: btnCrop.hovered
                visible: false
                onClicked: mainController.perform_cropping(true)
            }

            Button {
                id: btnUndo
                text: ""
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                icon.source: "../assets/icons/undo_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: Rectangle { color: "transparent"}
                ToolTip.text: "Undo crop"
                ToolTip.visible: btnUndo.hovered
                onClicked: mainController.undo_cropping(true)
                visible: false
            }
        }

        Rectangle {
            width: 1
            height: 24
            color: "#d0d0d0"
        }

        RowLayout {
            Layout.rightMargin: 5

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
                enabled: mainController.display_image()
                onCurrentIndexChanged: mainController.select_img_type(valueAt(currentIndex))
            }

            Button {
                id: btnShowGraph
                text: ""
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                icon.source: "../assets/icons/graph_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: Rectangle { color: "transparent"}
                ToolTip.text: "Show graph"
                ToolTip.visible: btnShowGraph.hovered
                onClicked: drpDownGraph.open()
                enabled: mainController.display_image()
                
                Popup {
                    id: drpDownGraph
                    width: 250
                    height: 480
                    modal: true
                    //focus: false
                    x: -250
                    y: 32
                    background: Rectangle {
                        color: "#f0f0f0"
                        border.color: "#d0d0d0"
                        border.width: 1
                        radius: 2
                    }
                    
                    ColumnLayout {
                        anchors.fill: parent

                        GraphExtractWidget{}
            
                        RowLayout {
                            spacing: 10
                            Layout.alignment: Qt.AlignHCenter
            
                            Button {
                                Layout.preferredWidth: 54
                                Layout.preferredHeight: 30
                                text: ""
                                onClicked: drpDownGraph.close()
            
                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: "#bc0000"
            
                                    Label {
                                        text: "Cancel"
                                        color: "#ffffff"
                                        anchors.centerIn: parent
                                    }
                                }
                            }
            
                            Button {
                                Layout.preferredWidth: 40
                                Layout.preferredHeight: 30
                                text: ""
                                onClicked: {
                                    mainController.run_extract_graph();
                                    drpDownGraph.close();
                                }
            
                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: "#22bc55"
            
                                    Label {
                                        text: "OK"
                                        color: "#ffffff"
                                        anchors.centerIn: parent
                                    }
                                }
                            }
                        }
                    }
                    
                }
                
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
            btnRescale.enabled = mainController.display_image();
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


