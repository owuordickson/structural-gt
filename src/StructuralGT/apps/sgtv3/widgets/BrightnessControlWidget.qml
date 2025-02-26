import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


Item {
    id: brightnessControl  // used for external access
    height: parent.height
    width: parent.width

    property int spbWidthSize: 170
    property int lblWidthSize: 100
    property int valueRole: Qt.UserRole + 4
    property alias clBrightnessCtrl: brightnessCtrlLayout

    ColumnLayout {
        id: brightnessCtrlLayout
        spacing: 10

        Repeater {
            id: brightnessCtrlRepeater
            model: imgControlModel
            delegate: RowLayout {
                objectName: "ctrlRowLayout"
                Layout.fillWidth: true
                Layout.leftMargin: 10
                Layout.alignment: Qt.AlignLeft

                Label {
                    id: label
                    Layout.preferredWidth: lblWidthSize
                    text: model.text
                }

                SpinBox {
                    id: spinBox
                    objectName: model.id
                    Layout.minimumWidth: spbWidthSize
                    Layout.fillWidth: true
                    editable: true
                    from: -100
                    to: 100
                    stepSize: 1
                    property var currVal: model.value
                    value: currVal
                    onValueChanged: updateValue(currVal, value)
                    onFocusChanged: updateValue(currVal, value)
                    onEditableChanged: updateValue(currVal, value)
                }

                function updateValue(curr_val, val) {
                    if (curr_val !== val){
                        curr_val = val;
                        var index = imgControlModel.index(model.index, 0);
                        imgControlModel.setData(index, val, valueRole);
                    }
                }

            }
        }

    }
}
