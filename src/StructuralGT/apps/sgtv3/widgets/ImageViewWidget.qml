import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs as QuickDialogs
import Qt.labs.platform as Platform

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
        visible: !mainController.display_image()

        ColumnLayout {
            anchors.centerIn: parent

            Label {
                id: lblWelcome
                //Layout.preferredWidth:
                text: "Welcome to Structural GT"
                color: "blue"
                //font.bold: true
                font.pixelSize: 24
            }

            RowLayout {
                //anchors.fill: parent

                ColumnLayout {

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

                }

                Rectangle {
                    Layout.leftMargin: 24
                    Layout.rightMargin: 12
                    width: 1
                    height: 75
                    color: "#c0c0c0"
                }

                ColumnLayout {

                    Label {
                        id: lblQuick
                        Layout.leftMargin: 5
                        //Layout.preferredWidth:
                        text: "Quick Analysis"
                        color: "#808080"
                        font.bold: true
                        font.pixelSize: 16
                    }

                    Button {
                        id: btnAddImage
                        Layout.preferredWidth: 125
                        Layout.preferredHeight: 32
                        text: ""
                        onClicked: fileDialog.open()

                        Rectangle {
                            anchors.fill: parent
                            radius: 5
                            color: "#808080"

                            Label {
                                text: "Add image"
                                color: "white"
                                font.bold: true
                                font.pixelSize: 12
                                anchors.centerIn: parent
                            }
                        }
                    }

                    Button {
                        id: btnAddImageFolder
                        Layout.preferredWidth: 125
                        Layout.preferredHeight: 32
                        text: ""
                        onClicked: folderDialog.open()

                        Rectangle {
                            anchors.fill: parent
                            radius: 5
                            color: "#808080"

                            Label {
                                text: "Add image folder"
                                color: "white"
                                font.bold: true
                                font.pixelSize: 12
                                anchors.centerIn: parent
                            }
                        }
                    }
                    //}

                }

            }

        }

    }


    Rectangle {
        id: imgContainer
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: "transparent"
        clip: true  // Ensures only the selected area is visible
        visible: mainController.display_image()

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
            source: ""
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
                    mainController.show_cropping_tool(false);
                } else {
                    mainController.show_cropping_tool(true);
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
        visible: mainController.display_image()


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

            Label {
                id: lblNavInfo
                //text: "1/1"
                text: ""
                Layout.alignment: Qt.AlignCenter
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


    Platform.FolderDialog {
        id: folderDialog
        title: "Select a Folder"
        onAccepted: {
            //console.log("Selected folder:", folder)
            mainController.add_multiple_images(folder);
        }
        //onRejected: {console.log("Canceled")}
    }

    QuickDialogs.FileDialog {
        id: fileDialog
        title: "Open file"
        nameFilters: ["Image files (*.jpg *.tif *.png *.jpeg)"]
        onAccepted: {
            //console.log("Selected file:", fileDialog.selectedFile)
            mainController.add_single_image(fileDialog.selectedFile);
        }
        //onRejected: console.log("File selection canceled")
    }


    function cropImage() {

        // Crop image through mainController
        mainController.crop_image(cropArea.x, cropArea.y, cropArea.width, cropArea.height);

        // Hide selection box
        cropArea.visible = false;
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgView.source = mainController.get_pixmap();
            welcomeContainer.visible = !mainController.display_image();
            imgContainer.visible = mainController.display_image();
            navControls.visible = mainController.display_image();
            zoomFactor = 1.0;
            lblNavInfo.text = mainController.get_img_nav_location()
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


