import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.platform
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
                    onClicked: folderDialog.open()
                }
            }

            Button {
                id: btnImportImages
                Layout.alignment: Qt.AlignLeft
                text: "Import image(s)"
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

    FolderDialog {
        id: folderDialog
        title: "Select a Folder"
        onAccepted: {
            //console.log("Selected folder:", folder)
            mainController.set_output_dir(folder)
        }
        onRejected: {console.log("Canceled")}
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            txtOutputDir.text = mainController.get_output_dir();
        }

    }

}
