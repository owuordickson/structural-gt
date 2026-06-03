import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic
import Theme 1.0

Item {
    id: graphComputationCtrl
    width: parent.width
    implicitHeight: gtComputationLayout.implicitHeight

    property int valueRole: Qt.UserRole + 4

    ColumnLayout {
        id: gtComputationLayout
        width: parent.width
        spacing: 2

        /*Label {
            wrapMode: Text.Wrap
            Layout.leftMargin: 15
            color: Theme.green
            font.pixelSize: 10
            Layout.preferredWidth: 200
            text: "**Note**: all these computations are applied on the giant graph ONLY."
        }*/

        Repeater {
            model: gtcListModel
            delegate: ColumnLayout {
                Layout.fillWidth: true
                spacing: 5

                Basic.CheckBox {
                    id: checkBox
                    Layout.leftMargin: 10
                    objectName: model.id
                    text: model.text
                    property bool isChecked: model.value === 1
                    checked: isChecked
                    onCheckedChanged: updateValue(isChecked, checked)
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

                    function updateValue(isChecked, checked) {
                        if (isChecked !== checked) {  // Only update if there is a change
                            //isChecked = checked
                            let val = checked ? 1 : 0;
                            let index = gtcListModel.index(model.index, 0);
                            gtcListModel.setData(index, val, valueRole);
                        }
                    }
                }

                // Dynamically load additional child content for specific IDs
                Loader {
                    id: childContentLoader
                    active: checkBox.checked
                    visible: active && item !== null
                    Layout.leftMargin: 20
                    sourceComponent: {
                        switch (model.id) {
                            case "display_ohms_histogram":
                                return ohmsComponent
                            case "compute_avg_node_connectivity":
                                return avgComponent
                            case "compute_persistent_homology":
                                return phComponent
                            case "compute_scaling_behavior":
                                return scalingComponent
                            default:
                                return null
                        }
                    }
                }
            }
        }
    }

    // Custom Component for 'display_ohms_histogram'
    Component {
        id: ohmsComponent
        ColumnLayout {
            MicroscopyPropertyWidget {
            }
        }
    }

    // Custom component for 'compute_scaling_behavior'
    Component {
        id: scalingComponent
        ColumnLayout {
            ScalingBehaviorWidget {
            }
        }
    }

    // Custom Component for 'compute_avg_node_connectivity'
    Component {
        id: avgComponent
        ColumnLayout {
            Label {
                wrapMode: Text.Wrap
                color: Theme.red
                font.pixelSize: 10
                Layout.preferredWidth: 200
                text: "**Warning**: this calculation takes long (esp. when node-count > 2000)"
            }
        }
    }

    // Custom Component for 'compute_persistent_homology'
    Component {
        id: phComponent
        ColumnLayout {
            Label {
                wrapMode: Text.Wrap
                color: Theme.red
                font.pixelSize: 10
                Layout.preferredWidth: 200
                text: "**Warning**: this calculation consumes a lot of RAM (esp. when node-count > 1000)"
            }
        }
    }
}