import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic


ColumnLayout {
    id: projectFoldersControls
    Layout.preferredHeight: 90
    Layout.preferredWidth: parent.width
    Layout.topMargin: 10
    Layout.leftMargin: 10
    Layout.rightMargin: 5
    spacing: 5


    RowLayout {
        id: rowLayoutProject
        visible: true//mainController.is_project_open()

        Label {
            text: "Project Name:"
            font.bold: true
        }

        Label {
            id: lblProjectName
            Layout.minimumWidth: 175
            Layout.fillWidth: true
            text: ""
            //elide: Text.ElideRight
        }
    }

    RowLayout {
        Label {
            text: "Output Dir:"
            font.bold: true
        }

        TextField {
            id: txtOutputDir
            Layout.minimumWidth: 175
            Layout.fillWidth: true
            text: ""
            //elide: Text.ElideRight
        }

        Basic.Button {
            id: btnChangeOutDir
            //text: "Change"
            icon.source: "../assets/icons/edit_icon.png"
            icon.width: 21
            icon.height: 21
            background: Rectangle {
                color: "transparent"
            }
            enabled: mainController.display_image()
            onClicked: outFolderDialog.open()
        }
    }

    Button {
        id: btnImportImages
        text: "Import image(s)"
        leftPadding: 10
        rightPadding: 10
        Layout.alignment: Qt.AlignHCenter
        enabled: mainController.display_image()
        onClicked: imageFileDialog.open()
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            txtOutputDir.text = mainController.get_output_dir();
            btnChangeOutDir.enabled = mainController.display_image();
            btnImportImages.enabled = mainController.display_image() || mainController.is_project_open();
        }

        function onProjectOpenedSignal(name) {
            lblProjectName.text = name;
            rowLayoutProject.visible = true;//mainController.is_project_open();
            btnImportImages.enabled = mainController.display_image() || mainController.is_project_open();
        }
    }
}