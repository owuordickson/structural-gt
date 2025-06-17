import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts

Rectangle {
    id: loggingDataContainer
    width: parent.width
    height: parent.height
    color: "transparent"

    ColumnLayout {
        anchors.fill: parent
        spacing: 5

        ScrollView {
            Layout.fillWidth: true
            Layout.fillHeight: true

            Basic.TextArea {
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

                function appendAndScroll(htmlText) {
                    lblTextLogs.append(htmlText);
                    lblTextLogs.cursorPosition = lblTextLogs.length;  // Auto-scroll to bottom
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
                lblTextLogs.appendAndScroll("<font color='blue'>" + val + "%: " + msg + "</font>");
            } else {
                lblTextLogs.appendAndScroll("<font color='blue'>" + msg + "</font>");
            }
        }

        function onErrorSignal(msg) {
            lblTextLogs.appendAndScroll("<font color='red'>" + msg + "</font>");
        }

        function onTaskTerminatedSignal(success_val, msg_data) {
            if (success_val) {
                lblTextLogs.appendAndScroll("<b><font color='#2222bc'>Task completed successfully!</font></b>");
            } else {
                lblTextLogs.appendAndScroll("<b><font color='#bc2222'>Task terminated due to an error. Try again.</font></b>");
            }

            if (msg_data.length >= 2) {
                lblTextLogs.appendAndScroll("<br><font color='gray'>" + msg_data[0] + "<br>" + msg_data[1] + "</font>");
            }
        }
    }
}