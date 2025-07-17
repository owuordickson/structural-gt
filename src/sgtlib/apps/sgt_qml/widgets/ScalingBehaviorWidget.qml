import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    id: scalingBehaviorContent
    Layout.preferredHeight: 140
    Layout.preferredWidth: parent.width - 75

    property int txtWidthSize: 70
    property int lblWidthSize: 80
    property int valueRole: Qt.UserRole + 4

    ColumnLayout {

        Repeater {
            model: gtcScalingModel
            delegate: RowLayout {

                Loader {
                    sourceComponent: {
                        if (model.id === "scaling_behavior_kernel_count" || model.id === "scaling_behavior_patches_per_kernel")
                            return txtComponent
                        else if (
                            model.id === "scaling_behavior_power_law_fit" ||
                            model.id === "scaling_behavior_truncated_power_law_fit" ||
                            model.id === "scaling_behavior_log_normal_fit"
                        )
                            if (model.id === "scaling_behavior_power_law_fit" ) {
                                return cbxComponent_w_Title
                            } else {
                                return cbxComponent
                            }
                        else
                            return null
                    }

                    Component {
                        id: cbxComponent
                        RowLayout {
                            CheckBox {
                                id: checkBox
                                objectName: model.id
                                text: model.text
                                //color: "#2222bc"
                                font.pixelSize: 10
                                Layout.leftMargin: 10
                                property bool isChecked: model.value === 1
                                checked: isChecked
                                onCheckedChanged: {
                                    if (isChecked !== checked) {  // Only update if there is a change
                                        isChecked = checked
                                        let val = checked ? 1 : 0;
                                        let index = gtcScalingModel.index(model.index, 0);
                                        gtcScalingModel.setData(index, val, valueRole);
                                    }
                                }
                            }
                        }
                    }

                    Component {
                        id: txtComponent
                        RowLayout {
                            Label {
                                id: label
                                wrapMode: Text.Wrap
                                color: "#2222bc"
                                font.pixelSize: 10
                                Layout.preferredWidth: lblWidthSize
                                Layout.leftMargin: 10
                                text: model.text
                            }

                            TextField {
                                id: txtField
                                objectName: model.id
                                color: "#2222bc"
                                font.pixelSize: 10
                                Layout.preferredWidth: txtWidthSize
                                text: model.value
                                onActiveFocusChanged: {
                                    if (focus) {
                                        btnOK.visible = true;
                                    }
                                }
                            }

                            Button {
                                id: btnOK
                                text: ""
                                Layout.preferredWidth: 40
                                Layout.preferredHeight: 26
                                Layout.rightMargin: 10
                                visible: false
                                onClicked: {
                                    btnOK.visible = false;

                                    let index = gtcScalingModel.index(model.index, 0);
                                    gtcScalingModel.setData(index, txtField.text, valueRole);
                                }

                                Rectangle {
                                    anchors.fill: parent
                                    radius: 5
                                    color: "#22bc55"

                                    Label {
                                        text: "OK"
                                        color: "#ffffff"
                                        //font.bold: true
                                        //font.pixelSize: 10
                                        anchors.centerIn: parent
                                    }
                                }
                            }
                        }
                    }

                    Component {
                        id: cbxComponent_w_Title
                        ColumnLayout {
                            Label {
                                wrapMode: Text.Wrap
                                //color: "#2222bc"
                                font.pixelSize: 12
                                font.bold: true
                                Layout.preferredWidth: 150
                                Layout.topMargin: 5
                                Layout.leftMargin: 10
                                text: "Select Curve Fit Model"
                            }

                            Loader {
                                sourceComponent: {cbxComponent}
                            }
                        }
                    }

                }
            }
        }
    }
}