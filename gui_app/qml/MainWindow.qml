import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: mainWindow
    width: 1024
    height: 800
    title: "GUI Tutorial"

    ColumnLayout {
        id: loginControlLayout
        anchors.centerIn: parent
        spacing: 10

        Label {
            id: lblName
            Layout.preferredWidth: 100
            text: "What is your name?"
        }

        TextField {
            id: txtName
            Layout.preferredWidth: 100
            text: ""
        }

        Button {
            id: btnOK
            text: "OK"
            onClicked: {
                lblName.text = "Welcome " + txtName.text;
                let msg = controller.process_name(txtName.text);
                console.log(msg)
            }
        }

    }


    Connections {
        target: controller

    }

}