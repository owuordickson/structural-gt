import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    Layout.fillWidth: true
    Layout.fillHeight: true

    ScrollView {
        width: parent.width
        height: parent.height

        GridLayout {
            anchors.fill: parent
            columns: 1

            //BrightnessControlWidget{}

        }

    }



}
