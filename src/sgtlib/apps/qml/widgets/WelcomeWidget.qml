import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts


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
            text: "StructuralGT: analyze microscopy images"
            color: "#2266ff"
            //font.bold: true
            font.pixelSize: 19
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
                    onClicked: dialogCreateProject.open()

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
                    onClicked: dialogFileProject.open()

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
                    onClicked: dialogFileImage.open()

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
                    onClicked: dialogFolderImage.open()

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


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            welcomeContainer.visible = imageController.display_image() ? false : !projectController.is_project_open();

        }

    }


    Connections {
        target: projectController

        function onProjectOpenedSignal(name) {
            welcomeContainer.visible = imageController.display_image() ? false : !projectController.is_project_open();
        }

    }

}


