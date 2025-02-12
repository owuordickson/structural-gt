import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

TableView {
    id: tableView
    Layout.fillWidth: true
    height: graphPropsTableModel.rowCount() * 30  // implicitHeight is 30    Layout.topMargin: 5
    Layout.leftMargin: 2
    Layout.rightMargin: 2
    model: graphPropsTableModel

    delegate: Rectangle {
        implicitWidth: column === 0 ? (tableView.width * 0.36) : (tableView.width * 0.64)
        implicitHeight: 30
        color: row % 2 === 0 ? "#f5f5f5" : "#ffffff" // Alternating colors

        Text {
            text: display
            wrapMode: Text.Wrap
            font.pixelSize: 10
            color: "#303030"
            anchors.fill: parent
            anchors.topMargin: 5
            anchors.leftMargin: 10
        }

        Loader {
            sourceComponent: column === 1 ? lineBorder : noBorder
        }
    }

    Component {
        id: lineBorder
        Rectangle {
            width: 1 // Border width
            height: 30
            color: "#e0e0e0" // Border color
            anchors.left: parent.left
        }
    }

    Component {
        id: noBorder
        Rectangle {
            width: 5 // Border width
            height: parent.height
            color: transientParent
            anchors.left: parent.left
        }
    }
}


