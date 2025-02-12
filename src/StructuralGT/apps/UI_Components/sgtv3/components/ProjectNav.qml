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
        //width: parent.width
        height: parent.height

        GridLayout {
            //anchors.fill: parent
            columns: 1

            //BrightnessControlWidget{}

        }
    }
    /*
    TreeView {
        id: treeView
        anchors.fill: parent
        //Layout.fillWidth: true
        //Layout.leftMargin: 10
        //Layout.alignment: Qt.AlignLeft
        model: extractModel

        delegate: Item {
            required property TreeView treeView
            required property int row
            required property int depth
            required property bool hasChildren
            required property bool expanded

            width: treeView.width
            height: 30

            RowLayout {
                spacing: 5
                anchors.fill: parent

                // Expand/Collapse Button
                Button {
                    visible: hasChildren
                    text: expanded ? "▼" : "▶"
                    onClicked: treeView.toggleExpanded(row)
                }

                // Display Item Name
                Label {
                    text: treeView.model.data(treeView.model.index(row, 0))
                }
            }
        }
    }
    */
}
