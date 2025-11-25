import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import Theme 1.0
import "../widgets"

Item {
    id: waitOverlay
    anchors.fill: parent
    visible: mainController.wait && !imageController.img_filters_busy
    z: 9999

    Rectangle {
        anchors.fill: parent
        color: Theme.semiTransparent // semi-transparent dark
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
            color: Theme.waitText
            horizontalAlignment: Text.AlignHCenter
            anchors.horizontalCenter: parent.horizontalCenter
        }
    }
}