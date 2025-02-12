import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

TreeView {
    id: treeView
    //anchors.fill: parent
    Layout.fillWidth: true
    Layout.leftMargin: 10
    Layout.alignment: Qt.AlignLeft
    model: gtModel

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
