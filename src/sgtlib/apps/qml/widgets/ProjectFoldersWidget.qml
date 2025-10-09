import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic


ColumnLayout {
    id: projectFoldersControls
    Layout.preferredHeight: 90
    Layout.preferredWidth: parent.width
    Layout.alignment: Qt.AlignTop
    Layout.topMargin: 10
    Layout.leftMargin: 10
    Layout.rightMargin: 5
    spacing: 5


    RowLayout {
        id: rowLayoutProject
        visible: projectController.is_project_open()

        Label {
            text: "Project Name:"
            font.bold: true
        }

        Text {
            id: txtProjectName
            Layout.minimumWidth: 175
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignLeft
            text: ""
            wrapMode: Text.NoWrap
            elide: Text.ElideRight
            maximumLineCount: 1        // ensures single-line behavior
            font.pixelSize: 10
            verticalAlignment: Text.AlignVCenter
            horizontalAlignment: Text.AlignLeft
            clip: true
        }

    }

    RowLayout {
        Label {
            text: "Output Dir:"
            font.bold: true
        }

        Rectangle {
            Layout.minimumWidth: 175
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignLeft
            implicitHeight: 24
            border.width: 1
            border.color: "#909090"
            radius: 5
            color: "transparent"

            Text {
                id: txtOutputDir
                anchors.fill: parent
                anchors.margins: 4
                text: ""
                wrapMode: Text.NoWrap
                elide: Text.ElideRight
                font.pixelSize: 10
                verticalAlignment: Text.AlignVCenter
                horizontalAlignment: Text.AlignLeft
                clip: true
            }
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
            enabled: imageController.display_image()
            onClicked: outFolderDialog.open()
        }
    }

    Button {
        id: btnImportImages
        text: "Import image(s)"
        leftPadding: 10
        rightPadding: 10
        Layout.alignment: Qt.AlignHCenter
        enabled: imageController.display_image()
        onClicked: imageFileDialog.open()
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            txtOutputDir.text = projectController.get_output_dir();
            btnChangeOutDir.enabled = imageController.display_image();
            btnImportImages.enabled = imageController.display_image() || projectController.is_project_open();
        }
    }

    Connections {
        target: projectController

        function onProjectOpenedSignal(name) {
            txtProjectName.text = name;
            rowLayoutProject.visible = projectController.is_project_open();
            btnImportImages.enabled = imageController.display_image() || projectController.is_project_open();
        }
    }
}