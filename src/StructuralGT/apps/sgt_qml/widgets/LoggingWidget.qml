import QtQuick
import QtQuick.Controls
import QtQuick.Layouts //1.15

Rectangle {
    id: loggingDataContainer
    width: parent.width
    height: parent.height
    color: "transparent"

    ColumnLayout {
        anchors.fill: parent
        spacing: 5
        padding: 5

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            TextArea {
                id: lblTextLogs
                wrapMode: Text.Wrap
                readOnly: true
                selectByMouse: true
                textFormat: TextEdit.RichText
                font.pixelSize: 10
                color: "#000000"
                background: Rectangle {
                    color: "white"
                    radius: 4
                }
            }
        }

        Button {
            text: "Clear Logs"
            Layout.alignment: Qt.AlignRight
            onClicked: lblTextLogs.text = ""
        }
    }

    Connections {
        target: mainController

        function onUpdateProgressSignal(val, msg) {
            if (val <= 100) {
                lblTextLogs.append("<font color='blue'>" + val + "%: " + msg + "</font>");
            } else {
                lblTextLogs.append("<font color='blue'>" + msg + "</font>");
            }
        }

        function onErrorSignal(msg) {
            lblTextLogs.append("<font color='red'>" + msg + "</font>");
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            if (success_val) {
                lblTextLogs.append("<b><font color='#2222bc'>Task completed successfully!</font></b>");
            } else {
                lblTextLogs.append("<b><font color='#bc2222'>Task terminated due to an error. Try again.</font></b>");
            }

            if (msg_data.length >= 2) {
                lblTextLogs.append("<br><font color='gray'>" + msg_data[0] + "<br>" + msg_data[1] + "</font>");
            }
        }
    }
}