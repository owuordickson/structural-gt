import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

TreeView {
    id: treeView
    model: gteTreeModel

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
                Layout.leftMargin: 10
                visible: hasChildren
                text: expanded ? "▼" : "▶"
                //text: expanded ? "∨" : ">"
                background: transientParent
                onClicked: treeView.toggleExpanded(row)
            }

            Loader {
                Layout.fillWidth: model.id === "remove_object_size" ?  false : true
                Layout.preferredWidth: 75
                Layout.leftMargin: hasChildren ? 0 : depth > 0 ? 50 : 10
                sourceComponent: model.id === "remove_object_size" ? txtFldComponent : cbxComponent
            }

            Component {
                id: cbxComponent

                CheckBox {
                    id: checkBox
                    //objectName: model.id
                    text: model.text
                    checked: model.value === 1
                    onClicked: {console.log(depth); console.log(row); console.log(model.text)}
                }
            }

            Component {
                id: txtFldComponent

                TextField {
                    id: txtField
                    //objectName: model.id
                    width: 80
                    text: model.value
                }
            }

        }
    }
}
