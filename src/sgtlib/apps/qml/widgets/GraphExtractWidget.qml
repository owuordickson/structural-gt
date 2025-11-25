import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic as Basic
import QtQuick.Layouts
import Theme 1.0

Item {
    id: gteTreeControl
    width: parent.width
    enabled: imageController.display_image() && imageController.enable_img_controls()

    property int treeViewHeight: 320
    property int treeViewWidth: 240
    property int idRole: (Qt.UserRole + 1)

    ColumnLayout {
        anchors.fill: parent

        TreeView {
            id: gteTreeView
            width: treeViewWidth
            height: treeViewHeight
            model: gteTreeModel

            ButtonGroup {
                id: btnGrpWeights
                exclusive: true
            }

            delegate: Item {
                required property TreeView treeView
                required property int row
                required property string id  // Ensure the id is passed for selection
                required property int depth
                required property bool hasChildren
                required property bool expanded

                implicitWidth: gteTreeView.width
                implicitHeight: 24

                RowLayout {
                    spacing: 5
                    anchors.fill: parent

                    // Expand/Collapse Button
                    Basic.Button {
                        id: btnExpand
                        //Layout.leftMargin: 10
                        //text: expanded ? "▼" : "▶"
                        visible: hasChildren
                        icon.source: expanded ? "../assets/icons/expand_down_icon.png" : "../assets/icons/expand_right_icon.png"
                        icon.width: 14
                        icon.height: 14
                        icon.color: enabled ? Theme.black : Theme.disabled
                        background: Rectangle {
                            color: "transparent"
                        }
                        onClicked: gteTreeView.toggleExpanded(row)
                    }

                    Loader {
                        Layout.fillWidth: (model.id !== "merge_node_radius_size" || model.id !== "prune_max_iteration_count" || model.id !== "remove_object_size")
                        Layout.preferredWidth: 75
                        Layout.leftMargin: hasChildren ? 0 : depth > 0 ? 50 : 10
                        sourceComponent: (model.id === "merge_node_radius_size" || model.id === "prune_max_iteration_count" || model.id === "remove_object_size")
                            ? txtFldComponent : model.text.startsWith("by")
                                ? rdoComponent : cbxComponent
                    }

                    Component {
                        id: cbxComponent

                        Basic.CheckBox {
                            id: checkBox
                            objectName: model.id
                            text: model.text
                            font.pixelSize: 11
                            property bool isChecked: model.value === 1
                            checked: isChecked
                            // Custom indicator
                            indicator: Rectangle {
                                width: 14
                                height: 14
                                radius: 3                       // slightly rounded corners
                                anchors.verticalCenter: parent.verticalCenter
                                border.color: checkBox.checked ? Theme.dodgerBlue : Theme.text
                                border.width: 2
                                color: "transparent"

                                // Inner "check mark"
                                Rectangle {
                                    visible: checkBox.checked
                                    width: 8
                                    height: 8
                                    radius: 1
                                    anchors.centerIn: parent
                                    color: Theme.darkGray
                                }
                            }
                            contentItem: Label {
                                text: checkBox.text
                                font: checkBox.font
                                color: Theme.text
                                verticalAlignment: Text.AlignVCenter
                                leftPadding: checkBox.indicator.width + 6
                            }
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

                        Basic.RadioButton {
                            id: rdoButton
                            objectName: model.id
                            text: model.text
                            font.pixelSize: 11
                            ButtonGroup.group: btnGrpWeights
                            checked: model.value
                            onClicked: btnGrpWeights.checkedButton = this
                            // Custom radio indicator
                            indicator: Rectangle {
                                width: 14
                                height: 14
                                radius: 7
                                y: (rdoButton.height - height) / 2   // center vertically
                                border.color: rdoButton.checked ? Theme.dodgerBlue : Theme.text
                                border.width: 2
                                color: "transparent"

                                Rectangle {
                                    visible: rdoButton.checked
                                    width: 8
                                    height: 8
                                    radius: 4
                                    anchors.centerIn: parent
                                    color: Theme.darkGray
                                }
                            }
                            contentItem: Label {
                                text: rdoButton.text
                                font: rdoButton.font
                                color: Theme.text
                                verticalAlignment: Text.AlignVCenter
                                leftPadding: rdoButton.indicator.width + 6
                            }
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
                                font.pixelSize: 11
                                objectName: model.id
                                width: 80
                                property int txtVal: model.value
                                text: txtVal
                            }

                            Button {
                                id: btnRemoveOk
                                text: ""
                                Layout.preferredWidth: 36
                                Layout.preferredHeight: 30
                                Layout.rightMargin: 10
                                onFocusChanged: {btnRemoveOk.visible = true;}
                                onClicked: {
                                    updateChild(model.id, txtField.text);
                                    btnRemoveOk.visible = false;
                                }

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: Theme.green

                                    Label {
                                        text: "OK"
                                        color: Theme.white
                                        //font.bold: true
                                        font.pixelSize: 10
                                        anchors.centerIn: parent
                                    }
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
                        for (let r = 0; r < rows; r++) {
                            let childIndex = gteTreeModel.index(r, 0, parentIndex);
                            let item_id = gteTreeModel.data(childIndex, idRole);
                            if (child_id === item_id) {
                                gteTreeModel.setData(childIndex, val, Qt.EditRole);
                            }
                        }
                    }
                }
            }

        }
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            gteTreeControl.enabled = imageController.display_image() && imageController.enable_img_controls();
        }

    }
}
