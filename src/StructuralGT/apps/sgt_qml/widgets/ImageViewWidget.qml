import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
//import QtQuick.Dialogs as QuickDialogs
//import Qt.labs.platform as Platform

ColumnLayout {
    Layout.fillWidth: true
    Layout.fillHeight: true
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

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
                        onClicked: createProjectDialog.open()

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
                        onClicked: projectFileDialog.open()

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
                        onClicked: imageFileDialog.open()

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
                        onClicked: imageFolderDialog.open()

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

        Flickable {
            id: flickableArea
            anchors.fill: parent
            contentWidth: imgView.width * imgView.scale
            contentHeight: imgView.height * imgView.scale
            clip: true
            flickableDirection: Flickable.HorizontalAndVerticalFlick

            ScrollBar.vertical: ScrollBar {
                id: vScrollBar
                policy: flickableArea.contentHeight > flickableArea.height ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
            }
            ScrollBar.horizontal: ScrollBar {
                id: hScrollBar
                policy: flickableArea.contentWidth > flickableArea.width ? ScrollBar.AlwaysOn : ScrollBar.AlwaysOff
            }

            Image {
                id: imgView
                width: flickableArea.width
                height: flickableArea.height
                anchors.centerIn: parent
                scale: zoomFactor
                transformOrigin: Item.Center
                fillMode: Image.PreserveAspectFit
                source: ""
            }
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
                onClicked: mainController.load_prev_image()
            }

            Label {
                id: lblNavInfo
                text: ""
                color: "#808080"
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
                onClicked: mainController.load_next_image()
            }

        }
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
            welcomeContainer.visible = mainController.display_image() ? false : !mainController.is_project_open();
            imgContainer.visible = mainController.display_image();
            navControls.visible = mainController.display_image();

            zoomFactor = 1.0;

            btnPrevious.enabled = mainController.enable_prev_nav_btn();
            btnNext.enabled = mainController.enable_next_nav_btn();
            lblNavInfo.text = mainController.get_img_nav_location();
            //console.log(src);
        }

        function onProjectOpenedSignal(name) {
            welcomeContainer.visible = mainController.display_image() ? false : !mainController.is_project_open();
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

        function onUpdateProgressSignal(val, msg) {
            if (val === 101) {
                lblNavInfo.text = msg;
            }
            btnNext.enabled = mainController.enable_next_nav_btn();
            lblNavInfo.text = mainController.get_img_nav_location();
        }

        function onTaskTerminatedSignal(success_val, msg_data){
            lblNavInfo.text = mainController.get_img_nav_location();
            btnNext.enabled = mainController.enable_next_nav_btn();
            lblNavInfo.text = mainController.get_img_nav_location();
        }

    }
}


