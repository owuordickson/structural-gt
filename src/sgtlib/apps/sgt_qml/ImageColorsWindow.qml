import QtQuick
import QtQuick.Controls
//import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import QtQuick.Window
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
            color: "transparent"
            visible: true

            RowLayout {
                spacing: 2
                Layout.margins: 10
                anchors.horizontalCenter: parent.horizontalCenter

                Button {
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
                    //visible: !mainController.processing_colors
                    //onClicked: mainController.()
                }

                Column {
                    //visible: mainController.processing_colors

                    SpinnerProgress {
                        //running: mainController.processing_colors
                        width: 24
                        height: 24
                    }
                }
            }
        }

        // Image Colors Layout (hidden if RetrieveButton is visible)
        Rectangle {
            id: colorsLayout
            Layout.fillHeight: true
            Layout.fillWidth: true
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
            imgSelectionControls.visible = true; // mainController.image_batches_exist() && mainController.is_img_3d();
            retrieveControls.visible = false;
            colorsLayout.visible = true;

            if (mainController.image_batches_exist() && mainController.is_img_3d()) {
                cbColorsBatchSelector.currentIndex = mainController.get_selected_img_batch();
                //cbColorsImageSelector.currentIndex = mainController.;
                //imgCurrent.source =
            } else {
                //imgCurrent.source =
            }
        }

        function onShowImageHistogramSignal(allow) {
            // Force refresh
            if (imgColorsWindow.visible) {
                imgColorsWindow.visible = allow;
            }
        }
    }

}