import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

/*TreeView {
    id: treeView
    //anchors.fill: parent
    Layout.fillWidth: true
    Layout.leftMargin: 10
    Layout.alignment: Qt.AlignLeft
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
}*/

TreeView {
        model: extractModel
        delegate: Item {
            width: parent.width
            height: childrenRect.height + 4
            Column {
                Text { text: model.data }
                Repeater {
                    model: model.children
                    delegate: Item {
                        x: 20
                        width: parent.width - 20
                        height: childrenRect.height + 4
                        Column {
                            Text { text: model.data }
                            // Recursive Repeater for nested levels
                            Repeater {
                                model: model.children
                                delegate: Item {
                                    x: 20
                                    width: parent.width - 20
                                    height: childrenRect.height + 4
                                    Column {
                                        Text { text: model.data }
                                        // Add more nested Repeater as needed
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
