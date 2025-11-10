import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts


Rectangle {
    id: imgContainer
    Layout.fillWidth: true
    Layout.fillHeight: true
    color: "transparent"
    clip: true  // Ensures only the selected area is visible
    visible: imageController.display_image()

    property real zoomFactor: 1.0
    property int selectedRole: (Qt.UserRole + 20)

    Flickable {
        id: flickableArea
        anchors.fill: parent
        contentWidth: imgView.width * imgView.scale
        contentHeight: imgView.height * imgView.scale
        //clip: true
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
            visible: !imageController.is_img_3d()
        }

        GridView {
            id: imgGridView
            width: flickableArea.width
            height: flickableArea.height
            anchors.centerIn: parent
            cellWidth: flickableArea.width * zoomFactor / 3
            cellHeight: flickableArea.height * zoomFactor / 3
            model: img3dGridModel
            visible: imageController.is_img_3d()

            delegate: Item {
                width: imgGridView.cellWidth
                height: imgGridView.cellHeight

                Rectangle {
                    width: parent.width - 2  // Adds horizontal spacing
                    height: parent.height - 2  // Adds vertical spacing
                    color: "#ffffff"  // Background color for spacing effect

                    Image {
                        source: model.image === "" ? "" : "data:image/png;base64," + model.image  // Base64 encoded image
                        width: parent.width
                        height: parent.height
                        anchors.centerIn: parent
                        //scale: zoomFactor
                        transformOrigin: Item.Center
                        fillMode: Image.PreserveAspectCrop
                        //cache: true
                    }

                    Label {
                        text: "Frame " + model.id
                        color: "#bc0022"
                        anchors.left: parent.left
                        anchors.top: parent.top
                        anchors.margins: 2
                        background: Rectangle {
                            color: "transparent"
                        }
                    }

                    CheckBox {
                        id: checkBox
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.margins: 2
                        property bool isSelected: model.selected === 1
                        checked: isSelected
                        onCheckedChanged: {
                            if (isSelected !== checked) {  // Only update if there is a change
                                isSelected = checked
                                let val = checked ? 1 : 0;
                                let index = img3dGridModel.index(model.index, 0);
                                img3dGridModel.setData(index, val, selectedRole);
                                imageController.toggle_selected_batch_image(model.id, isSelected);
                            }
                        }
                    }

                }
            }
        }

    }

    // Zoom controls
    Rectangle {
        id: zoomControls
        width: parent.width
        anchors.top: parent.top
        color: "transparent"
        visible: true

        RowLayout {
            anchors.fill: parent

            Basic.Button {
                id: btnZoomIn
                text: "+"
                Layout.preferredHeight: 24
                Layout.preferredWidth: 24
                Layout.alignment: Qt.AlignLeft
                Layout.margins: 5
                font.bold: true
                background: Rectangle {
                    color: "#80ffffff"
                }  // 80% opacity (50% transparency)
                ToolTip.text: "Zoom in"
                ToolTip.visible: btnZoomIn.hovered
                onClicked: zoomFactor = Math.min(zoomFactor + 0.1, 3.0) // Max zoom = 3x
            }

            Basic.Button {
                id: btnZoomOut
                text: "-"
                Layout.preferredHeight: 24
                Layout.preferredWidth: 24
                Layout.alignment: Qt.AlignRight
                Layout.margins: 5
                font.bold: true
                background: Rectangle {
                    color: "#80ffffff"
                }
                ToolTip.text: "Zoom out"
                ToolTip.visible: btnZoomOut.hovered
                onClicked: zoomFactor = Math.max(zoomFactor - 0.1, 0.5) // Min zoom = 0.5x
            }
        }
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgView.visible = !imageController.is_img_3d();
            imgGridView.visible = imageController.is_img_3d();
            imgContainer.visible = imageController.display_image();

            if (!imageController.is_img_3d()) {
                imgView.source = imageController.get_pixmap();
            } else {
                imgView.source = "";
            }
            zoomFactor = 1.0;
        }

    }


}


