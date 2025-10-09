import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic
import QtQuick.Controls.Fusion as Fusion
//import Qt5Compat.GraphicalEffects
import "../widgets"

// Icons retrieved from Iconfinder.com and used under the CC0 1.0 Universal Public Domain Dedication.

Rectangle {
    id: rectRibbon
    width: parent.width - 20
    height: 40
    radius: 5
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    border.width: 1

    property int modelValueRole: Qt.UserRole + 4

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

            Basic.Button {
                id: btnHideLeftPane
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
                    hidePane = !hidePane;
                    toggleLeftPane(hidePane);
                }
            }

        }
    }

    RowLayout {
        anchors.right: parent.right
        anchors.verticalCenter: parent.verticalCenter

        RowLayout {

            Basic.Button {
                id: btnRescale
                text: ""
                icon.source: "../assets/icons/rescale_icon.png" // Path to your icon
                icon.width: 20 // Adjust as needed
                icon.height: 20
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Re-scale large images"
                ToolTip.visible: btnRescale.hovered
                enabled: true
                onClicked: drpDownRescale.open()


                Popup {
                    id: drpDownRescale
                    width: 180
                    //height: colRadioButtons.implicitHeight + 10
                    height: 50
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

                        RowLayout {
                            id: allowScalingContainer
                            spacing: 2
                            //Layout.alignment: Qt.AlignHCenter
                            visible: !imageController.display_image()

                            Label {
                                text: "Auto Scale Image"
                                color: "#2266ff"
                            }

                            Switch {
                                id: toggleAllowScaling
                                checked: true
                                onCheckedChanged: {
                                    if (checked) {
                                        // Actions when switched on
                                        imageController.set_auto_scale(true)
                                    } else {
                                        // Actions when switched off
                                        imageController.set_auto_scale(false)
                                    }
                                }
                            }
                        }

                        RescaleControlWidget {
                        }

                    }

                }
            }

            Basic.Button {
                id: btnBrightness
                text: ""
                icon.source: "../assets/icons/brightness_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Adjust brightness/contrast"
                ToolTip.visible: btnBrightness.hovered
                onClicked: drpDownBrightness.open()
                enabled: imageController.display_image()

                Popup {
                    id: drpDownBrightness
                    width: 260
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

                    ColumnLayout {
                        anchors.fill: parent
                        BrightnessControlWidget {
                        }
                    }

                }

            }

            Basic.Button {
                id: btnSelect
                text: ""
                Layout.preferredWidth: 32
                Layout.preferredHeight: 32
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Select area to crop"
                ToolTip.visible: btnSelect.hovered
                visible: imageController.display_image()
                enabled: imageController.enable_img_controls()
                onClicked: enableRectangularSelect()

                Rectangle {
                    id: btnSelectBorder
                    width: 18
                    height: 18
                    //width: parent.width
                    //height: parent.height
                    anchors.centerIn: parent
                    radius: 2
                    color: "transparent"
                    border.width: 1
                    border.color: "black"
                }
            }

            Basic.Button {
                id: btnCrop
                text: ""
                icon.source: "../assets/icons/crop_icon.png" // Path to your icon
                icon.width: 21 // Adjust as needed
                icon.height: 21
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Crop to selection"
                ToolTip.visible: btnCrop.hovered
                visible: false
                onClicked: imageController.perform_cropping(true)
            }

            Basic.Button {
                id: btnUndo
                text: ""
                icon.source: "../assets/icons/undo_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Undo crop"
                ToolTip.visible: btnUndo.hovered
                onClicked: imageController.undo_applied_changes(true, "cropping", -1)
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

            Fusion.ComboBox {
                id: cbImageType
                Layout.minimumWidth: 150
                model: imgViewOptionModel
                currentIndex: 0
                textRole: "text"
                valueRole: "dataValue"

                implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                ToolTip.text: "Change image type"
                ToolTip.visible: hovered
                enabled: imageController.display_image()

                delegate: ItemDelegate {
                    id: control
                    width: cbImageType.width
                    text: model.text
                    enabled: model.visible === 1   // disable if visible == 0

                    // Keep default ComboBox visuals
                    font.bold:  index === cbImageType.currentIndex

                    // Optional: custom colors for hover/selection
                    background: Rectangle {
                        color: control.hovered ? "#5599ff" : "transparent"
                        radius: 4
                    }
                }

                // Fires only when the user selects a new option
                onActivated: (index) => {
                    // Update all to 0, only current to 1
                    for (let i = 0; i < imgViewOptionModel.rowCount(); ++i) {
                        let val = i === index ? 1 : 0;
                        let idx = imgViewOptionModel.index(i, 0)
                        imgViewOptionModel.setData(idx, val, modelValueRole);
                    }
                    // Call Python controller
                    imageController.apply_changes("");
                }
            }


            Basic.Button {
                id: btnShowGraph
                text: ""
                icon.source: "../assets/icons/graph_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: Rectangle {
                    color: "transparent"
                }
                ToolTip.text: "Extract graph"
                ToolTip.visible: btnShowGraph.hovered
                onClicked: drpDownGraph.open()
                enabled: imageController.display_image()

                Popup {
                    id: drpDownGraph
                    width: 250
                    height: 400
                    modal: true
                    focus: false
                    x: -225
                    y: 32
                    background: Rectangle {
                        color: "#f0f0f0"
                        border.color: "#d0d0d0"
                        border.width: 1
                        radius: 2
                    }

                    ColumnLayout {
                        anchors.fill: parent

                        GraphExtractWidget {
                        }

                        RowLayout {
                            spacing: 10
                            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

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
                                id: btnRunGraph
                                Layout.preferredWidth: 40
                                Layout.preferredHeight: 30
                                text: ""
                                visible: imageController.enable_img_controls()
                                onClicked: {
                                    drpDownGraph.close();
                                    graphController.run_extract_graph();
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
            imageController.enable_rectangular_selection(false)
            btnSelectBorder.border.color = "black"
            btnSelectBorder.enabled = false
        } else {
            imageController.enable_rectangular_selection(true)
            btnSelectBorder.border.color = "red"
            btnSelectBorder.enabled = true
        }
    }

    Connections {
        target: imageController

        function onShowCroppingToolSignal(allow) {
            btnCrop.visible = allow;
        }

        function onShowUnCroppingToolSignal(allow) {
            btnUndo.visible = allow;
        }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            btnRunGraph.visible = imageController.enable_img_controls();
            btnSelect.visible = imageController.display_image();
            allowScalingContainer.visible = !imageController.display_image();
            btnSelect.enabled = imageController.enable_img_controls();
            btnBrightness.enabled = imageController.display_image();
            cbImageType.enabled = imageController.display_image();
            btnShowGraph.enabled = imageController.display_image();

            drpDownRescale.height = imageController.display_image() ? 180 : 50;
            if (drpDownRescale.height === 180) {
                drpDownRescale.height = imageController.enable_img_controls() ? 180 : 0;
            }

            // Update the combobox view
            for (let i = 0; i < imgViewOptionModel.rowCount(); ++i) {
                let idx = imgViewOptionModel.index(i, 0);
                let itemVal = imgViewOptionModel.data(idx, modelValueRole);
                if (itemVal === 1) {
                    cbImageType.currentIndex = i;
                    break;
                }
            }
        }
    }

}


