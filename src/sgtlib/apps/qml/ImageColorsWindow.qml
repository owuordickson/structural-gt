import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls.Material as Material
import QtQuick.Controls.Fusion as Fusion
import "widgets"

Window {
    id: imgColorsWindow
    width: 768
    height: 720
    x: 1024  // Exactly starts where your app ends
    y: 100
    visible: false  // Only show when needed
    title: "Image Colors"

    property int valueRole: Qt.UserRole + 4

    ColumnLayout {
        anchors.fill: parent

        // Image Selection Layout
        Rectangle {
            id: imgSelectionControls
            height: 32
            Layout.topMargin: 10
            Layout.fillHeight: false
            Layout.fillWidth: true
            color: "transparent"
            visible: false

            RowLayout {
                spacing: 4
                anchors.centerIn: parent

                ComboBox {
                    id: cbColorsBatchSelector
                    Layout.minimumWidth: 75
                    model: imgBatchModel
                    implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                    textRole: "text"
                    valueRole: "value"
                    ToolTip.text: "Change image batch"
                    ToolTip.visible: cbColorsBatchSelector.hovered
                    onCurrentIndexChanged: mainController.select_img_batch(valueAt(currentIndex))
                }

                Rectangle {
                    id: vertColorsLine
                    width: 1
                    height: 18
                    color: "#d0d0d0"
                    visible: mainController.is_img_3d()
                }

                ComboBox {
                    id: cbColorsImageSelector
                    Layout.minimumWidth: 75
                    model: img3dGridModel
                    implicitContentWidthPolicy: ComboBox.WidestTextWhenCompleted
                    textRole: "text"
                    valueRole: "id"
                    ToolTip.text: "Select image"
                    ToolTip.visible: cbColorsImageSelector.hovered
                    currentIndex: 0
                    visible: mainController.is_img_3d()
                    onCurrentIndexChanged: {
                        mainController.reset_colors_model();
                        mainController.imageChangedSignal.emit();
                    }
                }

            }
        }

        // Retrieve button and spinner -- Layout (hidden if ImageColors is visible)
        Rectangle {
            id: retrieveControls
            height: 90
            Layout.fillHeight: false
            Layout.fillWidth: true
            Layout.topMargin: 5
            color: "transparent"
            visible: true

            ColumnLayout {
                spacing: 5
                anchors.centerIn: parent

                RowLayout {
                    spacing: 4
                    Layout.alignment: Qt.AlignHCenter
                    visible: !mainController.wait && !mainController.img_filters_busy

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
                        icon.source: "assets/icons/reload_icon.png"
                        icon.width: 21
                        icon.height: 21
                        icon.color: "transparent"   // important for PNGs
                        ToolTip.text: "Get the dominant colors of the image."
                        ToolTip.visible: btnGetColors.hovered
                        visible: !mainController.wait && !mainController.img_filters_busy
                        onClicked: {
                            let sel_img = cbColorsImageSelector.currentIndex;
                            let max_colors = spbMaxColors.value;
                            mainController.run_retrieve_img_colors(sel_img, max_colors);
                        }
                    }

                    Column {
                        visible: mainController.img_filters_busy

                        SpinnerProgress {
                            running: mainController.img_filters_busy
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
                                    border.color: "#d0d0d0"
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
                                    border.color: "#d0d0d0"
                                }
                            }
                        }

                        Fusion.Button {
                            id: btnEliminateColors
                            text: " Apply "
                            Layout.alignment: Qt.AlignHCenter
                            leftPadding: 15
                            rightPadding: 15
                            icon.source: "assets/icons/approve_icon.png"
                            icon.width: 21
                            icon.height: 21
                            icon.color: "transparent"   // important for PNGs
                            //ToolTip.text: "Eliminate the selected colors."
                            //ToolTip.visible: btnUndoEliminate.hovered
                            enabled: !mainController.img_filters_busy
                            onClicked: {
                                let sel_img = cbColorsImageSelector.currentIndex;
                                let swap_color = btnGrpSwap.checkedButton === rdoWhite ? 1 : 0;
                                mainController.run_eliminate_img_colors(sel_img, swap_color);
                            }
                        }

                        Fusion.Button {
                            id: btnUndoEliminate
                            text: " Undo "
                            Layout.alignment: Qt.AlignHCenter
                            leftPadding: 15
                            rightPadding: 15
                            icon.source: "assets/icons/undo_icon.png"
                            icon.width: 21
                            icon.height: 21
                            icon.color: "transparent"   // important for PNGs
                            //ToolTip.text: "Undo the last color elimination."
                            //ToolTip.visible: btnUndoEliminate.hovered
                            enabled: !mainController.img_filters_busy
                            onClicked: {
                                let sel_img = cbColorsImageSelector.currentIndex;
                                mainController.undo_applied_changes(true, "colors", sel_img)
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
            if (imgColorsWindow.visible) {
                imgSelectionControls.visible = mainController.image_batches_exist();
                vertColorsLine.visible = mainController.is_img_3d();
                cbColorsImageSelector.visible = mainController.is_img_3d();

                //retrieveControls.visible = imgColorsModel.rowCount() <= 0;
                colorsLayout.visible = imgColorsModel.rowCount() > 0;

                if (mainController.image_batches_exist()) {
                    cbColorsBatchSelector.currentIndex = mainController.get_selected_img_batch();
                    cbColorsImageSelector.currentIndex = 0;  // to fetch from mainController
                }

                let img_idx = cbColorsImageSelector.currentIndex;
                let base64_img = mainController.get_selected_image(img_idx, "original");
                if (base64_img !== "") {
                    imgCurrent.source = "data:image/png;base64," + base64_img;
                }
            }
        }

        function onShowImageFilterControls(allow) {
            // Force refresh
            if (imgColorsWindow.visible) {
                //mainController.imageChangedSignal.emit();
                imgColorsWindow.visible = allow;
            }
        }
    }

}