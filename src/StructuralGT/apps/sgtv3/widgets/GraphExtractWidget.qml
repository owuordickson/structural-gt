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
        id: btnGrpWeights
    }

    // Array to store selected IDs
    property var selectedIds: [] //['has_weights','merge_nearby_nodes','prune_dangling_edges','remove_disconnected_segments','remove_self_loops']
    property string selectedRdoId: "DIA"
    property int removeObjectSize: 500
    property int idRole: (Qt.UserRole + 1)
    property int textRole: (Qt.UserRole + 2)
    property int valueRole: (Qt.UserRole + 3)

    Component.onCompleted:  {
        if (selectedIds.length === 0) {  // Ensure it runs only once
            initializeSelectedIds();
        }
    }


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
                    ButtonGroup.group: btnGrpWeights
                    //checked: model.value === 1
                    checked: model.id === selectedRdoId  // Restore selection
                    onClicked: toggleSelectedRdoId(model.id)
                }
            }

            Component {
                id: txtFldComponent

                TextField {
                    id: txtField
                    objectName: model.id
                    width: 80
                    text: removeObjectSize
                    onEditingFinished: changeRemoveObjectSize(txtField.text)
                }
            }

        }
    }


    function changeRemoveObjectSize(value) {
        removeObjectSize = value;
        //console.log(value);
    }

    function toggleSelectedRdoId(id) {
        selectedRdoId = id;
        //console.log(id)
    }

    // Toggle selection in the selectedIds array
    function toggleSelection(id) {
        let index = selectedIds.indexOf(id);
        if (index === -1) {
            selectedIds.push(id);  // Add to selection
        } else {
            selectedIds.splice(index, 1);  // Remove from selection
        }
        //console.log(selectedIds)
    }

    // Restore previous selection after collapsing/expanding
    function restoreSelection() {
        selectionModel.clear();  // Clear current selection

        for (let row = 0; row < model.rowCount(); row++) {
            let index = model.index(row, 0);
            let item_id = model.data(index, idRole);  // IdRole
            if (selectedIds.includes(item_id)) {
                selectionModel.select(index, ItemSelectionModel.Select);  // Reselect previously selected items
            }
            //console.log(item_id)
        }
    }

    function initializeSelectedIds() {
        selectedIds = [];  // Reset in case it's called multiple times
        for (let row = 0; row < model.rowCount(); row++) {

            let index = model.index(row, 0);
            let item_id = model.data(index, idRole);  // IdRole
            let item_val = model.data(index, valueRole); // ValueRole

            if (item_val === 1) {  // Check if value is 1 (pre-selected)
                selectedIds.push(item_id);
            }
        }
        //console.log("Pre-selected IDs:", selectedIds);
    }

}
