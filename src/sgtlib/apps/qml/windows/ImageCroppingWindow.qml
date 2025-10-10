import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Window
import QtQuick.Controls.Material as Material
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

        const cropX = (cropTool.cropArea.x + offsetX) / scale;
        const cropY = (cropTool.cropArea.y + offsetY) / scale;
        const cropW = cropTool.cropArea.width / scale;
        const cropH = cropTool.cropArea.height / scale;

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
        cropTool.visible = false;
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