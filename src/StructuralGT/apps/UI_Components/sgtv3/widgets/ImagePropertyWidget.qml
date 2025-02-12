import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

TableView {
    id: tableView
    height: imgPropsTableModel.rowCount() * 30  // implicitHeight is 30
    //resizableColumns: true
    Layout.fillWidth: true
    Layout.topMargin: 5
    Layout.leftMargin: 2
    Layout.rightMargin: 2
    model: imgPropsTableModel

    delegate: Rectangle {
        implicitWidth: column === 0 ? (tableView.width * 0.36) : (tableView.width * 0.64) //imgPropsTableModel.columnCount
        implicitHeight: 30
        color: row % 2 === 0 ? "#f5f5f5" : "#ffffff" // Alternating colors
        //border.color: "#d0d0d0"
        //border.width: 1

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

    /*HorizontalHeaderView {
        syncView: tableView
        width: tableView.width
    }*/

    /*columnHeader: Rectangle {
        height: 40
        color: "lightgray"
        border.color: "#a0a0a0"

        Row {
            anchors.fill: parent
            Repeater {
                model: imgPropsTableModel.columnCount()
                delegate: Rectangle {
                    width: tableView.width / imgPropsTableModel.columnCount()
                    height: 40
                    border.color: "#c0c0c0"

                    Text {
                        anchors.centerIn: parent
                        text: imgPropsTableModel.headerData(index, Qt.Horizontal, Qt.DisplayRole)
                        font.bold: true
                    }
                }
            }
        }
    }*/

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


