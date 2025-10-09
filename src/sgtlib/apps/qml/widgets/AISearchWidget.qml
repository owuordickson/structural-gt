import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

RowLayout {
    Layout.preferredWidth: parent.width
    Layout.leftMargin: 10
    Layout.bottomMargin: 5
    visible: !aiController.ai_busy && aiController.ai_mode_active

    property int valueRole: Qt.UserRole + 4

    Repeater {
        model: aiSearchModel
        delegate: CheckBox {
            id: checkBox
            objectName: model.id
            font.pixelSize: 11
            text: model.text
            visible: model.visible === 1
            ToolTip.text: model.tooltip
            ToolTip.visible: checkBox.hovered
            property bool isChecked: model.value === 1
            checked: isChecked
            onCheckedChanged: updateValue(isChecked, checked)

            function updateValue(isChecked, checked) {
                if (isChecked !== checked) {  // Only update if there is a change
                    let val = checked ? 1 : 0;
                    let index = aiSearchModel.index(model.index, 0);
                    aiSearchModel.setData(index, val, valueRole);
                }
            }
        }
    }
}