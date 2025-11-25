import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0
import "../widgets"

Rectangle {
    width: parent.width - 20
    height: parent.height - 10
    color: Theme.background

    GridLayout {
        anchors.fill: parent
        columns: 1

        ImageView{}

    }

}
