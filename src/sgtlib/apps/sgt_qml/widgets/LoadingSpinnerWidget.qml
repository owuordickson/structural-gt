import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic

Item {
    id: waitOverlay
    anchors.fill: parent
    visible: mainController.wait
    z: 9999

    Rectangle {
        anchors.fill: parent
        color: "#80000000" // semi-transparent dark
    }

    Column {
        anchors.centerIn: parent
        spacing: 12

        SpinnerProgress{
            running: mainController.wait
            width: 64
            height: 64
        }

        Label {
            text: mainController.wait_text
            font.pointSize: 21
            color: "#2299ff"
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
        }
    }
}