import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls.Basic as Basic
import "../widgets"


Window {
    id: imgCroppingWindow
    width: 768
    height: 720
    x: 1024  // Exactly starts where your app ends
    y: 40
    //flags: Qt.Window | Qt.FramelessWindowHint
    visible: false  // Only show when needed
    title: "Crop Image"

    ColumnLayout {
        anchors.fill: parent

        //  Cropping Controls
        Rectangle {
            id: rectcropWidgets

            RowLayout {

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
                    //visible: false
                    onClicked: {
                        imageController.perform_cropping(true);
                        toggleRectangularSelect();
                    }
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
                    onClicked: {
                        imageController.undo_applied_changes(true, "cropping", -1);
                        toggleRectangularSelect();
                    }
                    //visible: false
                }

                Button {
                    id: btnSave
                    text: "Save Image"
                    //icon.source: "../assets/icons/save_icon.png"
                    icon.width: 24
                    icon.height: 24
                    //background: Rectangle { color: "transparent" }
                    ToolTip.text: "Save Image"
                    ToolTip.visible: btnSave.hovered
                    //onClicked:
                    //visible: false
                }
            }
        }

        // Image View
        Rectangle {
            id: rectImageContainer

            Image {
            }

            // Cropping
            CroppingWidget {
                id: cropWidget
            }
        }

        // Image Navigation Controls
        ImageNavControls {
            id: imgNavControls
        }

    }

    function getActualImageSize() {
        const containerWidth = flickableArea.width;
        const containerHeight = flickableArea.height;

        const imageSourceWidth = imgView.sourceSize.width;
        const imageSourceHeight = imgView.sourceSize.height;

        if (imageSourceWidth <= 0 || imageSourceHeight <= 0)
            return {width: 0, height: 0};

        const imgAspect = imageSourceWidth / imageSourceHeight;
        const containerAspect = containerWidth / containerHeight;

        let actualWidth, actualHeight;
        if (imgAspect > containerAspect) {
            // Image is wider than container, so width fits
            actualWidth = containerWidth;
            actualHeight = containerWidth / imgAspect;
        } else {
            // Image is taller than container, so height fits
            actualHeight = containerHeight;
            actualWidth = containerHeight * imgAspect;
        }

        return {width: actualWidth, height: actualHeight};
    }

    function getCropAreaInImageCoords() {
        const scale = zoomFactor;
        const offsetX = flickableArea.contentX;
        const offsetY = flickableArea.contentY;
        const actualSize = getActualImageSize();

        const cropX = (cropWidget.cropArea.x + offsetX) / scale;
        const cropY = (cropWidget.cropArea.y + offsetY) / scale;
        const cropW = cropWidget.cropArea.width / scale;
        const cropH = cropWidget.cropArea.height / scale;

        return {
            x: Math.round(cropX),
            y: Math.round(cropY),
            width: Math.round(cropW),
            height: Math.round(cropH),
            actualWidth: Math.round(actualSize.width),
            actualHeight: Math.round(actualSize.height)
        };
    }

    function cropImage() {

        // Crop image through Controller
        const cropRect = getCropAreaInImageCoords();
        imageController.crop_image(cropRect.x, cropRect.y, cropRect.width, cropRect.height, cropRect.actualWidth, cropRect.actualHeight);

        // Hide the selection box
        cropWidget.visible = false;
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
        target: imageController

        function onPerformCroppingSignal(allow) {
            if (allow) {
                cropImage();
            }
        }

    }
}