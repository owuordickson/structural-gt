import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Basic as Basic
import Theme 1.0

RowLayout {
    Layout.preferredWidth: parent.width
    Layout.leftMargin: 10
    Layout.bottomMargin: 5
    visible: !aiController.ai_busy && aiController.ai_mode_active

    property int valueRole: Qt.UserRole + 4

    Repeater {
        model: aiSearchModel
        delegate: Basic.CheckBox {
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
                color: Theme.darkGrey
                verticalAlignment: Text.AlignVCenter
                leftPadding: checkBox.indicator.width + 6
            }

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