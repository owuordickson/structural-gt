import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0
import "../widgets"

Dialog {
    id: createProjectDialog
    anchors.centerIn: parent
    title: "Create SGT Project"
    modal: true
    width: 300
    height: 200

    ColumnLayout {
        anchors.fill: parent
        CreateProjectWidget {
            id: createProjectControls
        }

        RowLayout {
            spacing: 10
            //Layout.topMargin: 10
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

            Button {
                Layout.preferredWidth: 54
                Layout.preferredHeight: 30
                text: ""
                onClicked: createProjectDialog.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.errorColor

                    Label {
                        text: "Cancel"
                        color: "#ffffff"
                        anchors.centerIn: parent
                    }
                }
            }

            Button {
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                onClicked: {
                    var name = createProjectControls.txtName.text;
                    var location = createProjectControls.txtLocation.text;

                    if (name === "") {
                        //console.log("Please fill in all fields.");
                        createProjectControls.lblName.text = "Name*";
                        createProjectControls.lblName.color = "red";
                        createProjectControls.txtName.placeholderText = "please enter a name!"

                    } else if (location === "") {
                        createProjectControls.lblLocation.text = "Location*";
                        createProjectControls.lblLocation.color = "red";

                    } else {
                        createProjectDialog.close();
                        projectController.create_sgt_project(name, location);
                    }
                }

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.successColor

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