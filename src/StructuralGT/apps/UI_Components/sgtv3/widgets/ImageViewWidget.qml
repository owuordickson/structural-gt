import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


ColumnLayout {
    Layout.fillWidth: true
    Layout.fillHeight: true
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

    // Zoom Factor Variable
    property real zoomFactor: 1.0

    Rectangle {
        id: welcomeContainer
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: "transparent"
        visible: !imageController.is_image_loaded()

        ColumnLayout {
            anchors.centerIn: parent

            Label {
                id: imgLabel
                text: "Welcome to Structural GT"
                color: "blue"
                //font.bold: true
                font.pixelSize: 24
            }

            Button {
                id: btnCreateProject
                Layout.preferredWidth: 180
                Layout.preferredHeight: 48
                text: ""

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: "yellow"

                    Label {
                        text: "Create project..."
                        color: "#808080"
                        font.bold: true
                        font.pixelSize: 16
                        anchors.centerIn: parent
                    }
                }
            }

            Button {
                id: btnOpenProject
                Layout.preferredWidth: 180
                Layout.preferredHeight: 48
                background: transientParent
                text: ""

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: "transparent"
                    border.width: 2
                    border.color: "#808080"

                    Label {
                        text: "Open project..."
                        color: "#808080"
                        font.bold: true
                        font.pixelSize: 16
                        anchors.centerIn: parent
                    }
                }
            }

            /*Rectangle {
                width: 150
                height: 2
                color: "lightgray"
            }*/


            /*Button {
                id: btnAddImageFolder
                text: "Add image folder"
            }

            Button {
                id: btnAddImage
                text: "Add image"
            }*/

        }

    }


    Rectangle {
        id: imgContainer
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: "transparent"
        clip: true  // Ensures only the selected area is visible
        visible: imageController.is_image_loaded()

        /*ScrollView {
            width: parent.width
            height: parent.height*/

        Image {
            id: imgView
            //width: parent.width
            //height: parent.height
            //anchors.centerIn: parent
            anchors.fill: parent
            scale: zoomFactor
            transformOrigin: Item.Center
            fillMode: Image.PreserveAspectFit
            source: imageController.get_pixmap()
        }

        //}

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

        Rectangle {
            id: zoomControls
            width: parent.width
            anchors.top: parent.top
            color: "transparent"
            visible: true

            RowLayout {
                anchors.fill: parent

                Button {
                    id: btnZoomIn
                    text: "+"
                    Layout.alignment: Qt.AlignLeft
                    ToolTip.text: "Zoom in"
                    ToolTip.visible: btnZoomIn.hovered
                    onClicked: zoomFactor = Math.min(zoomFactor + 0.1, 3.0) // Max zoom = 3x
                }

                Button {
                    id: btnZoomOut
                    text: "-"
                    Layout.alignment: Qt.AlignRight
                    ToolTip.text: "Zoom out"
                    ToolTip.visible: btnZoomOut.hovered
                    onClicked: zoomFactor = Math.max(zoomFactor - 0.1, 0.5) // Min zoom = 0.5x
                }
            }
        }
    }

    Rectangle {
        id: navControls
        height: 32
        Layout.fillHeight: false
        Layout.fillWidth: true
        color: "transparent"
        enabled: false
        visible: imageController.is_image_loaded()


        RowLayout {
            anchors.fill: parent

            Button {
                id: btnPrevious
                text: ""
                icon.source: "../assets/icons/back_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: transientParent
                Layout.alignment: Qt.AlignLeft
                //Layout.margins: 5
            }

            Button {
                id: btnNext
                text: ""
                icon.source: "../assets/icons/next_icon.png" // Path to your icon
                icon.width: 24 // Adjust as needed
                icon.height: 24
                background: transientParent
                Layout.alignment: Qt.AlignRight
                //Layout.margins: 5
            }

        }
    }


    // Save Button
    //imageProcessor.adjust_brightness_contrast(parseFloat(brightnessInput.text), parseFloat(contrastInput.text));
    /*Button {
        text: "Save Processed Image"
        anchors.horizontalCenter: parent
        onClicked: {
            colorMatrix.grabToImage(function(result) {
                if (result && result.image) {
                    imageProcessor.save_image(result.image);
                }
            });
        }
    }*/

    function cropImage() {

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
            imgView.source = imageController.get_pixmap(); // Force refresh
            zoomFactor = 1.0
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

        function onAdjustBrightnessContrastSignal(b_val, c_val) {
            imgView.grabToImage(function(result) {
                if (result && result.image) {  // Ensure result is valid
                    imageController.adjust_brightness_contrast(result.image, b_val, c_val);
                }
            });
        }

    }
}


