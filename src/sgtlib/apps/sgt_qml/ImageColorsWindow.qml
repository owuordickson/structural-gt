import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls.Material as Material
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
        Rectangle {
            id: imgSelectionControls
            height: 48
            Layout.topMargin: 5
            Layout.fillHeight: false
            Layout.fillWidth: true
            color: "transparent"
            visible: false

            RowLayout {
                spacing: 4
                anchors.centerIn: parent
                anchors.verticalCenter: parent.verticalCenter

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
                    onCurrentIndexChanged: mainController.imageChangedSignal.emit()
                }

            }
        }

        // Retrieve button and spinner -- Layout (hidden if ImageColors is visible)
        Rectangle {
            id: retrieveControls
            Layout.fillHeight: true
            Layout.fillWidth: true
            Layout.topMargin: 5
            color: "transparent"
            visible: true

            ColumnLayout {
                spacing: 10
                anchors.centerIn: parent

                RowLayout {
                    spacing: 4
                    anchors.horizontalCenter: parent.horizontalCenter
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
                    anchors.horizontalCenter: parent.horizontalCenter

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
                    width: 600
                    height: 600
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
                    height: 560
                    color: "transparent"

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 5

                        ListView {
                            id: colorList
                            Layout.fillWidth: true
                            Layout.fillHeight: true   // take remaining space
                            clip: true
                            model: imgColorsModel

                            delegate: RowLayout {
                                width: ListView.view.width   // full width
                                height: 32                   // fixed height for consistency
                                spacing: 4

                                CheckBox {
                                    id: checkBox
                                    objectName: model.id
                                    text: model.text
                                    property bool isChecked: model.value
                                    checked: isChecked
                                }
                            }
                        }

                        Material.Button {
                            id: btnEliminateColors
                            Layout.alignment: Qt.AlignHCenter
                            leftPadding: 10
                            rightPadding: 10
                            text: "Apply Changes"
                            enabled: true
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

                retrieveControls.visible = imgColorsModel.rowCount() <= 0;
                colorsLayout.visible = imgColorsModel.rowCount() > 0

                if (mainController.image_batches_exist()) {
                    cbColorsBatchSelector.currentIndex = mainController.get_selected_img_batch();
                    cbColorsImageSelector.currentIndex = 0;
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