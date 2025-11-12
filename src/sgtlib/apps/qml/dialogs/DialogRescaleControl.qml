import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Dialog {
    id: dialogRescaleCtrl
    //parent: mainWindow
    anchors.centerIn: parent
    title: "Re-scale Image"
    modal: true
    width: 250
    height: 240

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
                onClicked: dialogRescaleCtrl.close()

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