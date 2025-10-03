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
                    width: 1
                    height: 18
                    color: "#d0d0d0"
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
                spacing: 8
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
                spacing: 2
                Layout.margins: 5
                Layout.fillWidth: true
                anchors.horizontalCenter: parent.horizontalCenter

                Image {
                    id: imgCurrent
                    width: 650
                    height: 650
                    anchors.centerIn: parent
                    transformOrigin: Item.Center
                    fillMode: Image.PreserveAspectFit
                    source: ""
                }

                Rectangle {
                    width: 118
                    height: 650
                    Layout.leftMargin: 5
                    color: "gray"

                }
            }
        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            imgSelectionControls.visible = mainController.image_batches_exist() && mainController.is_img_3d();
            retrieveControls.visible = true;
            colorsLayout.visible = false;

            if (mainController.image_batches_exist() && mainController.is_img_3d()) {
                cbColorsBatchSelector.currentIndex = mainController.get_selected_img_batch();
                //cbColorsImageSelector.currentIndex = mainController.;
                //imgCurrent.source =
            } else {
                //imgCurrent.source =
            }
        }

        function onShowImageFilterControls(allow) {
            // Force refresh
            if (imgColorsWindow.visible) {
                imgColorsWindow.visible = allow;
            }
        }
    }

}