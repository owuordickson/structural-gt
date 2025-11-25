import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0
import "../widgets"

Dialog {
    id: dialogRescaleCtrl
    //parent: mainWindow
    anchors.centerIn: parent
    title: "Re-scale Image"
    modal: true
    width: 250
    height: 240
    background: Rectangle {color: Theme.background}

    ColumnLayout {
        anchors.fill: parent
        RescaleControlWidget {
            id: rescaleControls
        }
        //rescaleControls.lblScale.visible: false

        RowLayout {
            spacing: 10
            //Layout.topMargin: 10
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

            Button {
                Layout.preferredWidth: 54
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogRescaleCtrl.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.red

                    Label {
                        text: "Cancel"
                        color: Theme.white
                        anchors.centerIn: parent
                    }
                }
            }

            Button {
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogRescaleCtrl.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.green

                    Label {
                        text: "OK"
                        color: Theme.white
                        anchors.centerIn: parent
                    }
                }
            }
        }

    }
}