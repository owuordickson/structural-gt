import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
//import TableModel

TableView {
    id: tableView
    anchors.fill: parent
    model: imgPropsTableModel

    delegate: Rectangle {
        implicitWidth: 80
        implicitHeight: 25
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
