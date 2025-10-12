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
    title: "Cropping Image"

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        //  Cropping Controls
        Rectangle {
            id: rectCropControls
            height: 36
            Layout.fillHeight: false
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignTop | Qt.AlignHCenter
            Layout.topMargin: 10
            Layout.leftMargin: 10
            Layout.rightMargin: 10
            radius: 5
            color: "transparent"
            visible: true

            RowLayout {
                anchors.verticalCenter: parent.verticalCenter
                anchors.horizontalCenter: parent.horizontalCenter
                spacing: 5

                Basic.Button {
                    id: btnCrop
                    text: ""
                    //Layout.alignment: Qt.AlignHCenter
                    icon.source: "../assets/icons/crop_icon.png" // Path to your icon
                    icon.width: 21 // Adjust as needed
                    icon.height: 21
                    background: Rectangle {
                        color: "transparent"
                    }
                    ToolTip.text: "Crop to selection"
                    ToolTip.visible: btnCrop.hovered
                    visible: false
                    onClicked: cropImage()
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
                    }
                    visible: false
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

        // x, y, width, height coordinates (editable)

        // Image View
        Rectangle {
            id: rectImageContainer
            Layout.fillHeight: true
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            Layout.margins: 20
            color: "transparent"

            Image {
                id: imgCrop
                width: parent.width
                height: parent.height
                anchors.centerIn: parent
                transformOrigin: Item.Center
                fillMode: Image.PreserveAspectCrop
                source: ""
            }

            // Cropping
            CroppingWidget {
                id: cropWidget
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

    function getActualImageSize() {
        const containerWidth = rectImageContainer.width;
        const containerHeight = rectImageContainer.height;

        const imageSourceWidth = imgCrop.sourceSize.width;
        const imageSourceHeight = imgCrop.sourceSize.height;

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
        const scale = 1.0;
        const offsetX = 0;
        const offsetY = 0;
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

    function loadImage() {
        let img_pos = imgNavControls.img_pos;
        let base64_img = imageController.get_selected_image(img_pos, "original");
        imgCrop.source = "data:image/png;base64," + base64_img;
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            //if (imgCroppingWindow.visible) {
                loadImage();
            //}
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
}