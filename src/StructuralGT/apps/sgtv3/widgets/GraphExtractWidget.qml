import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

TreeView {
    id: treeView
    model: gteTreeModel
    selectionModel: ItemSelectionModel { model: treeView.model }
    onExpanded: {
        restoreSelection();  // Restore selection when expanding
    }
    onCollapsed: {
        restoreSelection();  // Restore selection when expanding
    }

    ButtonGroup {
        id: btnGroup
    }

    // Array to store selected IDs
    property var selectedIds: []
    ListModel { id: selectedIdsModel } // Store selected IDs


    delegate: Item {
        required property TreeView treeView
        required property int row
        required property string id  // Ensure the id is passed for selection
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
                sourceComponent: model.id === "remove_object_size" ? txtFldComponent :
                                                                     model.text.startsWith("by") ? rdoComponent : cbxComponent
            }

            Component {
                id: cbxComponent

                CheckBox {
                    id: checkBox
                    objectName: model.id
                    text: model.text
                    //checked: model.value === 1
                    //onClicked: {console.log(model.text)}
                    checked: selectedIds.includes(model.id)  // Restore selection
                    onClicked: toggleSelection(model.id)
                }
            }

            Component {
                id: rdoComponent

                RadioButton {
                    id: rdoButton
                    objectName: model.id
                    text: model.text
                    ButtonGroup.group: btnGroup
                    //checked: model.value === 1
                    //onClicked: {console.log(model.text)}
                    checked: selectedIds.includes(model.id)  // Restore selection
                    onClicked: toggleSelection(model.id)
                }
            }

            Component {
                id: txtFldComponent

                TextField {
                    id: txtField
                    objectName: model.id
                    width: 80
                    text: model.value
                }
            }

        }
    }


    // Toggle selection in the selectedIds array
        function toggleSelection(id) {
            let index = selectedIds.indexOf(id);
            if (index === -1) {
                selectedIds.push(id);  // Add to selection
            } else {
                selectedIds.splice(index, 1);  // Remove from selection
            }
        }

        // Restore previous selection after collapsing/expanding
        function restoreSelection() {
            selectionModel.clear();  // Clear current selection

            for (let row = 0; row < treeView.model.rowCount(); row++) {
                let index = treeView.model.index(row, 0);
                let item = treeView.model.data(index, gteTreeModel.IdRole);  // Access the ID with custom role
                console.log(index)
                console.log(item)
                if (selectedIds.includes(item)) {
                    selectionModel.select(index, ItemSelectionModel.Select);  // Reselect previously selected items
                }
            }
        }

}
