import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    id: loggingDataContainer
    width: parent.width
    height: parent.height

    property string logText: ""

    ColumnLayout {
        anchors.fill: parent
        spacing: 5
        padding: 5

        // Consider using TextArea for scrollable, long logs
        TextArea {
            id: lblTextLogs
            Layout.fillWidth: true
            Layout.fillHeight: true
            readOnly: true
            wrapMode: Text.Wrap
            font.pixelSize: 10
            color: "blue"
            text: logText
        }
    }


    Connections {
        target: mainController

        function onUpdateProgressSignal(val, msg) {
            if (val <= 100) {
                logText = logText + val + '%' + ': ' + msg+ '\n' ;
            } else {
                logText = logText + msg + '\n' ;
            }
        }

        function onErrorSignal (msg) {
            logText = logText + msg + '\n' ;
        }

        function onTaskTerminatedSignal(success_val, msg_data){
            if (success_val) {
                lblTextLogs.color = "#2222bc";
                logText = logText + 'Task completed successfully!' + '\n' ;
            } else {
                lblTextLogs.color = "#bc2222";
                logText = logText + 'Task terminated due to an error. Try again.' + '\n' ;
            }

            if (msg_data.length > 0) {
                logText = logText + '\n\n' + msg_data[0] + '\n' + msg_data[1] + '\n' ;
                lblTextLogs.color = success_val ? "#2222bc" : "#bc2222";
            }
        }
    }
}