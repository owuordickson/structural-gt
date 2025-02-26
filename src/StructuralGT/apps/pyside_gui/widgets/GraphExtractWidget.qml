import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

TreeView {
    id: treeView
    enabled: mainController.display_image()
    model: gteTreeModel

    ButtonGroup {
        id: btnGrpWeights
        exclusive: true
    }

    property int idRole: (Qt.UserRole + 1)
    property int textRole: (Qt.UserRole + 2)
    property int valueRole: (Qt.UserRole + 3)

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
                    property bool isChecked: model.value === 1 ? true : false
                    checked: isChecked
                    onCheckedChanged: {
                        if (isChecked !== checked) {  // Only update if there is a change
                            isChecked = checked
                            let val = checked ? 1 : 0;
                            var index = gteTreeModel.index(model.index, 0);
                            gteTreeModel.setData(index, val, Qt.EditRole);
                        }
                    }
                }
            }

            Component {
                id: rdoComponent

                RadioButton {
                    id: rdoButton
                    objectName: model.id
                    text: model.text
                    ButtonGroup.group: btnGrpWeights
                    checked: model.value === 1 ? true :  false
                    onClicked: btnGrpWeights.checkedButton = this
                    onCheckedChanged: {
                        var val = checked ? 1 : 0;
                        updateChild(model.id, val);
                    }
                }
            }

            Component {
                id: txtFldComponent

                RowLayout {

                    TextField {
                        id: txtField
                        objectName: model.id
                        width: 80
                        property int txtVal: model.value
                        text: txtVal
                    }

                    Button {
                        id: btnRemoveOk
                        text: "OK"
                        onClicked: {
                            updateChild(model.id, txtField.text);
                            btnRemoveOk.visible = false;
                        }
                        onFocusChanged: {btnRemoveOk.visible = true;}
                    }

                }
            }

        }
    }

    function updateChild(child_id, val) {
        let row_count = gteTreeModel.rowCount();
        for (let row = 0; row < row_count; row++) {
            let parentIndex = gteTreeModel.index(row, 0);
            let rows = gteTreeModel.rowCount(parentIndex);
            for (let r = 0; r < rows; r++){
                let childIndex = gteTreeModel.index(r, 0, parentIndex);
                let item_id = gteTreeModel.data(childIndex, idRole);
                if (child_id === item_id) {
                    gteTreeModel.setData(childIndex, val, Qt.EditRole);
                }
            }
        }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            treeView.enabled = mainController.display_image();
        }

    }
}
