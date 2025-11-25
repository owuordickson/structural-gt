import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls.Material as Material
import QtQuick.Controls.Fusion as Fusion
import Theme 1.0
import "../widgets"

Window {
    id: imgColorsWindow
    width: 768
    height: 720
    x: 1024  // Exactly starts where your app ends
    y: 100
    visible: false  // Only show when needed
    title: "Image Colors"

    property int valueRole: Qt.UserRole + 4
    property int selectedRole: Qt.UserRole + 20


    ColumnLayout {
        anchors.fill: parent

        // Retrieve button and spinner -- Layout (hidden if ImageColors is visible)
        Rectangle {
            id: retrieveControls
            height: 90
            Layout.fillHeight: false
            Layout.fillWidth: true
            Layout.topMargin: 10
            color: "transparent"
            visible: true

            ColumnLayout {
                spacing: 5
                anchors.centerIn: parent

                RowLayout {
                    spacing: 4
                    Layout.alignment: Qt.AlignHCenter
                    visible: !mainController.wait && !imageController.img_filters_busy

                    Material.Label {
                        text: "Maximum Unique Colors: "
                        font.pixelSize: 14
                    }

                    Material.SpinBox {
                        id: spbMaxColors
                        from: 2
                        to: 256
                        stepSize: 1
                        value: 10
                        editable: true

                        font.pixelSize: 10
                        implicitWidth: 75
                        implicitHeight: 28
                        //Layout.preferredWidth: 75 // if inside RowLayout
                        //Layout.preferredHeight: 28
                    }
                }

                RowLayout {
                    spacing: 2
                    Layout.alignment: Qt.AlignHCenter

                    Material.Button {
                        id: btnGetColors
                        text: " Retrieve Colors"
                        leftPadding: 10
                        rightPadding: 10
                        icon.source: "../assets/icons/reload_icon.png"
                        icon.width: 21
                        icon.height: 21
                        icon.color: "transparent"   // important for PNGs
                        ToolTip.text: "Get the dominant colors of the image."
                        ToolTip.visible: btnGetColors.hovered
                        visible: !mainController.wait && !imageController.img_filters_busy
                        onClicked: {
                            let img_pos = imgNavControls.img_pos;
                            let max_colors = spbMaxColors.value;
                            imageController.run_retrieve_img_colors(img_pos, max_colors);
                        }
                    }

                    Label {
                        text: "Another task is running, please wait..."
                        color: Theme.errorColor
                        visible: mainController.wait && !imageController.img_filters_busy
                    }

                    Column {
                        visible: imageController.img_filters_busy

                        SpinnerProgress {
                            running: imageController.img_filters_busy
                            width: 24
                            height: 24
                        }

                        Label {
                            text: "please wait..."
                            font.pointSize: 12
                            color: "#2299ff"
                            horizontalAlignment: Text.AlignHCenter
                            anchors.horizontalCenter: parent.horizontalCenter
                        }
                    }
                }
            }
        }

        // Image Colors Layout (hidden if RetrieveButton is visible)
        Rectangle {
            id: colorsLayout
            Layout.fillHeight: true
            Layout.fillWidth: true
            Layout.topMargin: 5
            color: "transparent"
            visible: false

            RowLayout {
                spacing: 10
                anchors.centerIn: parent

                Rectangle {
                    width: 512
                    height: 512
                    color: "transparent"

                    Image {
                        id: imgCurrent
                        width: parent.width
                        height: parent.height
                        anchors.centerIn: parent
                        transformOrigin: Item.Center
                        fillMode: Image.PreserveAspectCrop
                        source: ""
                    }
                }

                Rectangle {
                    width: 120
                    height: 480
                    color: "transparent"

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 5

                        Label {
                            text: "Dominant colors in the image:"
                            color: "#606060"
                            wrapMode: Text.Wrap
                            font.pixelSize: 10
                        }

                        ListView {
                            id: colorList
                            Layout.fillWidth: true
                            Layout.fillHeight: true   // take remaining space
                            clip: true
                            model: imgColorsModel

                            delegate: RowLayout {
                                width: ListView.view.width   // full width
                                height: 32                   // fixed height for consistency
                                spacing: 2

                                CheckBox {
                                    id: checkBox
                                    objectName: model.id
                                    property bool isChecked: model.value
                                    checked: isChecked
                                    onCheckedChanged: {
                                        if (isChecked !== checked) {  // Only update if there is a change
                                            isChecked = checked
                                            let val = checked ? 1 : 0;
                                            let index = imgColorsModel.index(model.index, 0);
                                            imgColorsModel.setData(index, val, valueRole);
                                        }
                                    }
                                }

                                // Color swatch instead of hex string
                                Rectangle {
                                    width: 75
                                    height: 24
                                    radius: 2
                                    color: model.text   // assuming model.text holds "#RRGGBB"
                                }

                                // Optional: show the hex code as the tooltip, not as text
                                ToolTip.visible: checkBox.hovered
                                ToolTip.text: model.text
                            }
                        }

                        Label {
                            Layout.alignment: Qt.AlignHCenter
                            text: "Swap to:"
                            color: "#606060"
                            wrapMode: Text.Wrap
                            font.pixelSize: 10
                        }

                        Row {
                            Layout.alignment: Qt.AlignHCenter
                            Layout.bottomMargin: 15
                            spacing: 4

                            ButtonGroup {
                                id: btnGrpSwap
                                checkedButton: rdoBlack
                                exclusive: true
                            }

                            RowLayout {
                                height: 28
                                spacing: 2

                                RadioButton {
                                    id: rdoBlack
                                    Layout.alignment: Qt.AlignVCenter
                                    ButtonGroup.group: btnGrpSwap
                                    onClicked: btnGrpSwap.checkedButton = this
                                }

                                Rectangle {
                                    Layout.alignment: Qt.AlignVCenter
                                    width: 25
                                    height: 25
                                    radius: 2
                                    color: "#000000"
                                    border.width: 1
                                    border.color: Theme.borderColor
                                }
                            }

                            RowLayout {
                                height: 28
                                spacing: 2

                                RadioButton {
                                    id: rdoWhite
                                    Layout.alignment: Qt.AlignVCenter
                                    ButtonGroup.group: btnGrpSwap
                                    onClicked: btnGrpSwap.checkedButton = this
                                }

                                Rectangle {
                                    Layout.alignment: Qt.AlignVCenter
                                    width: 25
                                    height: 25
                                    radius: 2
                                    color: "#ffffff"
                                    border.width: 1
                                    border.color: Theme.borderColor
                                }
                            }
                        }

                        Fusion.Button {
                            id: btnEliminateColors
                            text: " Apply "
                            Layout.alignment: Qt.AlignHCenter
                            leftPadding: 15
                            rightPadding: 15
                            icon.source: "../assets/icons/approve_icon.png"
                            icon.width: 21
                            icon.height: 21
                            icon.color: "transparent"   // important for PNGs
                            //ToolTip.text: "Eliminate the selected colors."
                            //ToolTip.visible: btnUndoEliminate.hovered
                            enabled: !imageController.img_filters_busy
                            onClicked: {
                                let img_pos = imgNavControls.img_pos;
                                let swap_color = btnGrpSwap.checkedButton === rdoWhite ? 1 : 0;
                                imageController.run_eliminate_img_colors(img_pos, swap_color);
                            }
                        }

                        Fusion.Button {
                            id: btnUndoEliminate
                            text: " Undo "
                            Layout.alignment: Qt.AlignHCenter
                            leftPadding: 15
                            rightPadding: 15
                            icon.source: "../assets/icons/undo_icon.png"
                            icon.width: 21
                            icon.height: 21
                            icon.color: "transparent"   // important for PNGs
                            //ToolTip.text: "Undo the last color elimination."
                            //ToolTip.visible: btnUndoEliminate.hovered
                            enabled: !imageController.img_filters_busy
                            onClicked: {
                                let img_pos = imgNavControls.img_pos;
                                imageController.undo_applied_changes(true, "colors", img_pos)
                            }
                        }
                    }

                }
            }
        }

        // Image Navigation Controls
        ImageNavControls {
            id: imgNavControls
            showPrev: false
            showNext: false
            showImgBatch: false
            showImgPos: true
        }

    }

    function loadColorsData() {
        colorsLayout.visible = imgColorsModel.rowCount() > 0;

        let img_pos = imgNavControls.img_pos;
        let base64_img = imageController.get_selected_image(img_pos, "mutated");
        imgCurrent.source = "data:image/png;base64," + base64_img;
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            if (imgColorsWindow.visible) {
                loadColorsData();
            }
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            if (imgColorsWindow.visible) {
                loadColorsData();
            }
        }
    }

}