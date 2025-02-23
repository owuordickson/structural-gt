import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs as QuickDialogs
import Qt.labs.platform as Platform
import "../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    Layout.fillWidth: true
    Layout.fillHeight: true

    ScrollView {
        //width: parent.width
        height: parent.height

        ColumnLayout {
            //anchors.fill: parent

            RowLayout {
                Layout.topMargin: 10
                Layout.leftMargin: 10
                Layout.bottomMargin: 5

                Label {
                    text: "Output Dir:"
                    font.bold: true
                }

                TextField {
                    id: txtOutputDir
                    Layout.minimumWidth: 175
                    Layout.fillWidth: true
                    text: ""
                }

                Button {
                    id: btnChangeOutDir
                    //text: "Change"
                    icon.source: "../assets/icons/edit_icon.png"
                    icon.width: 21
                    icon.height: 21
                    background: transientParent
                    enabled: mainController.display_image();
                    onClicked: folderDialog.open()
                }
            }

            Button {
                id: btnImportImages
                Layout.alignment: Qt.AlignLeft
                text: "Import image(s)"
                enabled: mainController.display_image();
                onClicked: fileDialog.open()
            }

            Rectangle {
                height: 1
                color: "#d0d0d0"
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 5
                Layout.leftMargin: 20
                Layout.rightMargin: 20
            }
            Text {
                text: "Image List"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 5
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
                visible: true
            }

            ProjectWidget {}
        }
    }

    Platform.FolderDialog {
        id: folderDialog
        title: "Select a Folder"
        onAccepted: {
            //console.log("Selected folder:", folder)
            mainController.set_output_dir(folder)
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


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            txtOutputDir.text = mainController.get_output_dir();
            btnChangeOutDir.enabled = mainController.display_image();
            btnImportImages.enabled = mainController.display_image();
        }

    }

}
