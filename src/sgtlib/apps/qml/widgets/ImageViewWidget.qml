import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts


ColumnLayout {
    Layout.fillWidth: true
    Layout.fillHeight: true
    Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

    property real zoomFactor: 1.0
    property int selectedRole: (Qt.UserRole + 20)

    Rectangle {
        id: welcomeContainer
        Layout.fillWidth: true
        Layout.fillHeight: true
        color: "transparent"
        visible: !imageController.display_image()

        ColumnLayout {
            anchors.centerIn: parent

            Label {
                id: lblWelcome
                //Layout.preferredWidth:
                text: "Welcome to StructuralGT"
                color: "#2266ff"
                //font.bold: true
                font.pixelSize: 24
            }

            RowLayout {
                //anchors.fill: parent

                ColumnLayout {

                    Basic.Button {
                        id: btnCreateProject
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: 48
                        background: Rectangle {
                            color: "transparent"
                        }
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

                    Basic.Button {
                        id: btnOpenProject
                        Layout.preferredWidth: 180
                        Layout.preferredHeight: 48
                        background: Rectangle {
                            color: "transparent"
                        }
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
        id: imgViewControls
        height: 32
        Layout.fillHeight: false
        Layout.fillWidth: true
        color: "transparent"
        visible: imageController.display_image()

        RowLayout {
            anchors.verticalCenter: parent.verticalCenter
            anchors.horizontalCenter: parent.horizontalCenter

            Switch {
                id: toggleShowGiantGraph
                visible: graphController.display_graph()
                text: "only giant"
                ToolTip.text: "Display only the giant graph network."
                ToolTip.visible: toggleShowGiantGraph.hovered
                checked: false // Initial state
                onCheckedChanged: {
                    if (checked) {
                        // Actions when switched on
                        graphController.reload_graph_image(true);
                    } else {
                        // Actions when switched off
                        graphController.reload_graph_image(false);
                    }
                }
            }

            Button {
                id: btnLoad3DGraph
                leftPadding: 10
                rightPadding: 10
                text: " view"
                icon.source: "../assets/icons/3d_icon.png"
                icon.width: 21
                icon.height: 21
                icon.color: "transparent"   // important for PNGs
                ToolTip.text: "Load OVITO 3D graph visualization."
                ToolTip.visible: btnLoad3DGraph.hovered
                visible: graphController.display_graph()
                onClicked: graphController.load_graph_simulation()
            }

            Button {
                id: btnGraphRating
                leftPadding: 10
                rightPadding: 10
                text: " rate"
                icon.source: "../assets/icons/thumbs-up-emoji.png"
                icon.width: 21
                icon.height: 21
                icon.color: "transparent"   // important for PNGs
                ToolTip.text: "How good is the graph? Give a score..."
                ToolTip.visible: btnGraphRating.hovered
                visible: graphController.display_graph()
                onClicked: drpDownRating.open()

                Popup {
                    id: drpDownRating
                    width: 400
                    height: 172
                    modal: true
                    focus: false
                    x: -225
                    y: 32
                    background: Rectangle {
                        color: "#f0f0f0"
                        border.color: "#d0d0d0"
                        border.width: 1
                        radius: 2
                    }

                    ColumnLayout {
                        anchors.fill: parent
                        spacing: 2

                        GraphRatingWidget{
                            id: graphRating
                        }

                        RowLayout {
                            spacing: 10
                            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

                            Button {
                                Layout.preferredWidth: 54
                                Layout.preferredHeight: 30
                                text: ""
                                onClicked: drpDownRating.close()

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: "#bc0000"

                                    Label {
                                        text: "Cancel"
                                        color: "#ffffff"
                                        anchors.centerIn: parent
                                    }
                                }
                            }

                            Button {
                                id: btnSendRating
                                Layout.preferredWidth: 40
                                Layout.preferredHeight: 30
                                text: ""
                                visible: imageController.enable_img_controls()
                                onClicked: {
                                    drpDownRating.close();
                                    graphController.rate_graph(graphRating.rating);
                                }

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: "#22bc55"

                                    Label {
                                        text: "OK"
                                        color: "#ffffff"
                                        anchors.centerIn: parent
                                    }
                                }
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
        visible: imageController.display_image()

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
    }

    // Image Navigation Controls
    ImageNavControls{}


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgView.visible = !imageController.is_img_3d();
            imgGridView.visible = imageController.is_img_3d();
            welcomeContainer.visible = imageController.display_image() ? false : !projectController.is_project_open();
            imgContainer.visible = imageController.display_image();
            imgViewControls.visible = imageController.display_image();
            toggleShowGiantGraph.visible = graphController.display_graph();
            btnLoad3DGraph.visible = graphController.display_graph();
            btnGraphRating.visible = graphController.display_graph();
            btnSendRating.visible = imageController.enable_img_controls();

            if (!imageController.is_img_3d()) {
                imgView.source = imageController.get_pixmap();
            } else {
                imgView.source = "";
            }
            zoomFactor = 1.0;
        }

    }


    Connections {
        target: projectController

        function onProjectOpenedSignal(name) {
            welcomeContainer.visible = imageController.display_image() ? false : !projectController.is_project_open();
        }

    }

}


