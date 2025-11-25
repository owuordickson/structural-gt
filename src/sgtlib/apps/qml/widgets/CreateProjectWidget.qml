import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Qt.labs.platform as Platform
import Theme 1.0


Item {
    //height: 75//parent.height
    //width: parent.width
    Layout.preferredHeight: 100
    Layout.preferredWidth: parent.width

    // Expose TextFields as properties to make them accessible
    property alias lblName: lblProjectName
    property alias lblLocation: lblProjectLocation

    property alias txtName: txtProjectName
    property alias txtLocation: txtProjectLocation

    property int txtWidthSize: 200
    property int lblWidthSize: 64


    ColumnLayout {
        spacing: 10

        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 10
            Layout.alignment: Qt.AlignLeft

            Label {
                id: lblProjectName
                Layout.preferredWidth: lblWidthSize
                text: "Name:"
                color: Theme.text
                //font.bold: true
            }

            TextField {
                id: txtProjectName
                Layout.minimumWidth: txtWidthSize
                Layout.fillWidth: true
                text: ""
                onEditingFinished: {
                    lblProjectName.text = "Name:"
                    lblProjectName.color = "black";
                    txtProjectName.placeholderText = "";
                }
            }
        }

        RowLayout {
            Layout.fillWidth: true
            Layout.leftMargin: 10
            Layout.alignment: Qt.AlignLeft

            Label {
                id: lblProjectLocation
                Layout.preferredWidth: lblWidthSize
                text: "Location:"
                color: Theme.text
                //font.bold: true
            }

            TextField {
                id: txtProjectLocation
                Layout.minimumWidth: txtWidthSize
                Layout.fillWidth: true
                readOnly: true
                text: ""
                placeholderText: "click here to select folder..."

                MouseArea {
                    anchors.fill: parent
                    onClicked: {
                        onClicked: projectFolderDialog.open()
                    }
                }
            }
        }

    }


    Platform.FolderDialog {
        id: projectFolderDialog
        title: "Select a project location"
        onAccepted: {
            lblLocation.text = "Location:"
            lblLocation.color = Theme.black;

            txtProjectLocation.text = projectFolderDialog.folder;
        }
        //onRejected: {console.log("Canceled")}
    }

}
