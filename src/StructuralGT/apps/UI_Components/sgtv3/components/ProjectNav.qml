import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    Layout.fillWidth: true
    Layout.fillHeight: true

    /*ScrollView {
        //width: parent.width
        height: parent.height

        GridLayout {
            //anchors.fill: parent
            columns: 1

            //BrightnessControlWidget{}

        }
    }*/

    TreeView {
        id: treeView
        anchors.fill: parent
        model: graphTreeModel

        delegate: Item {
            required property TreeView treeView
            required property int row
            required property int depth
            required property bool hasChildren
            required property bool expanded

            implicitWidth: treeView.width
            implicitHeight: 24

            RowLayout {
                spacing: 5
                anchors.fill: parent

                // Expand/Collapse Button
                Button {
                    visible: hasChildren
                    text: expanded ? "▼" : "▶"
                    //text: expanded ? "∨" : ">"
                    background: transientParent
                    onClicked: treeView.toggleExpanded(row)
                }

                // Display Item Name
                Label {
                    text: model.display  // Correct way to access model data
                    elide: Label.ElideRight
                    Layout.fillWidth: true
                }

            }
        }
    }

}
