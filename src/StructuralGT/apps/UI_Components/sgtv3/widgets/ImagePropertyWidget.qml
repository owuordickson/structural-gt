import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
//import TableModel

TableView {
    id: tableView
    Layout.fillWidth: true
    Layout.fillHeight: true
    model: imgPropsTableModel

    delegate: Rectangle {
        implicitWidth: tableView.width / 2
        implicitHeight: 20
        border.color: "#d0d0d0"
        border.width: 1


        Text {
            text: display
            anchors.centerIn: parent
        }
    }

    // Headers
    /*headerHorizontal:  Rectangle {
        implicitWidth: tableView.width
        implicitHeight: 30
        color: "lightgray"

        Row {
            anchors.fill: parent
            Repeater {
                model: tableView.columnCount
                delegate: Rectangle {
                    width: tableView.width / tableView.columnCount
                    height: 30

                    Text {
                        anchors.centerIn: parent
                        text: tableView.model.headerData(index, Qt.Horizontal, Qt.DisplayRole)
                    }
                }
            }
        }
    }*/
}
