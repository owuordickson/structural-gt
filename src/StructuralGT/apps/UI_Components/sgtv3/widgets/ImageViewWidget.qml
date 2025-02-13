import QtQuick
import QtQuick.Controls
import QtQuick.Layouts



ColumnLayout {
    //width: 512
    //height: 512
    Layout.fillWidth: true
    Layout.fillHeight: true
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
    //anchors.fill: parent
    //spacing: 10

    Label {
        id: imgLabel
        text: "No image loaded"
        visible: !imageController.is_image_loaded()
    }

    Rectangle {
        id: imgContainer
        width: 300
        height: 300
        //anchors.centerIn: parent
        clip: true  // Ensures only the selected area is visible
        border.color: "black"  // TO DELETE
        border.width: 2        // TO DELETE
        visible: imageController.is_image_loaded()

        Image {
            id: imgView
            width: parent.width
            height: parent.height
            //Layout.fillWidth: true
            //Layout.fillHeight: true
            fillMode: Image.PreserveAspectFit
            source: imageController.get_pixmap()
        }

        // Selection Rectangle for Cropping
        Rectangle {
            id: cropArea
            color: "transparent"
            border.color: "red"
            border.width: 2
            visible: false

            // Draggable functionality
            MouseArea {
                id: dragArea
                anchors.fill: parent
                drag.target: cropArea
                drag.minimumX: 0
                drag.minimumY: 0
                drag.maximumX: imgContainer.width - cropArea.width
                drag.maximumY: imgContainer.height - cropArea.height
            }
        }

        MouseArea {
            id: selectionArea
            anchors.fill: parent
            enabled: false
            onPressed: (mouse) => {
                           cropArea.x = mouse.x;
                           cropArea.y = mouse.y;
                           cropArea.width = 0;
                           cropArea.height = 0;
                           cropArea.visible = true;
                       }
            onPositionChanged: (mouse) => {
                                   if (cropArea.visible) {
                                       cropArea.width = Math.abs(mouse.x - cropArea.x);
                                       cropArea.height = Math.abs(mouse.y - cropArea.y);
                                   }
                               }
            onReleased: {
                if (cropArea.width < 5 || cropArea.height < 5) {
                    cropArea.visible = false;  // Hide small selections
                    imageController.show_cropping_tool(false);
                } else {
                    imageController.show_cropping_tool(true);
                }
            }
        }

    }

    function cropImage() {
        /*
        // Move the main image so that only the selection area is visible
        imgView.x = cropArea.x;
        imgView.y = cropArea.y;
        imgView.width = cropArea.width;
        imgView.height = cropArea.height;

        // Resize the container to match the selection
        imgContainer.width = cropArea.width;
        imgContainer.height = cropArea.height;*/

        // Crop image through ImageController
        imgView.grabToImage(function(result) {
            if (result && result.image) {  // Ensure result is valid
                imageController.crop_image(result.image, cropArea.x, cropArea.y, cropArea.width, cropArea.height);
            }
        });

        // Hide selection box
        cropArea.visible = false;
    }

    Connections {
        target: imageController

        function onImageChangedSignal(src, newPath) {
            imgView.source = ""
            imgView.source = imageController.get_pixmap(); // Force refresh
            //console.log(src);
        }

        function onEnableRectangularSelectionSignal(allow) {
            if (allow) {
                selectionArea.enabled = true;
                cropArea.visible = true
            } else {
                selectionArea.enabled = false
                cropArea.visible = false
            }
        }

        function onPerformCroppingSignal(allow) {
            if (allow) {
                cropImage();
            }
        }
    }
}


