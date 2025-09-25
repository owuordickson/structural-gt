import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "widgets"

Rectangle {
    width: parent.width - 20
    height: parent.height - 10
    color: "#f0f0f0"

    GridLayout {
        anchors.fill: parent
        columns: 1

        ImageViewWidget{}

    }

}
