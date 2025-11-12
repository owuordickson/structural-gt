import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: dialogBrightnessCtrl
    //parent: mainWindow
    anchors.centerIn: parent
    title: "Control Brightness/Contrast"
    modal: true
    width: 260
    height: 150

    ColumnLayout {
        anchors.fill: parent
        BrightnessControlWidget {
        }

        RowLayout {
            spacing: 10
            //Layout.topMargin: 10
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

            Button {
                Layout.preferredWidth: 54
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogBrightnessCtrl.close()

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
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogBrightnessCtrl.close()

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