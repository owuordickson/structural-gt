import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


Item {
    id: graphComputationCtrl
    Layout.preferredWidth: parent.width
    Layout.alignment: Qt.AlignTop


    property int valueRole: Qt.UserRole + 4

    ColumnLayout {
        id: gtComputationLayout
        spacing: 10

        Repeater {
            model: gtcListModel
            delegate: ColumnLayout {
                Layout.fillWidth: true
                spacing: 5

                /*RowLayout {
                    Layout.fillWidth: true
                    Layout.leftMargin: 10*/

                    CheckBox {
                        Layout.leftMargin: 10
                        id: checkBox
                        objectName: model.id
                        text: model.text
                        property bool isChecked: model.value === 1
                        checked: isChecked
                        onCheckedChanged: updateValue(isChecked, checked)

                        function updateValue(isChecked, checked) {
                            if (isChecked !== checked) {  // Only update if there is a change
                                isChecked = checked
                                let val = checked ? 1 : 0;
                                var index = gtcListModel.index(model.index, 0);
                                gtcListModel.setData(index, val, valueRole);
                                //console.log(model.id)
                            }
                        }
                    }
                //}

                ColumnLayout {
                    id: childContent
                    visible: checkBox.checked && model.id === "display_ohms_histogram"
                    Layout.leftMargin: 20

                    MicroscopyPropertyWidget{}
                }
            }

        }

    }

}
