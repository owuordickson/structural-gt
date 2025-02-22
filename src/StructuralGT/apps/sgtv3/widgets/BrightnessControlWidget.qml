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
                    value: model.value
                    onValueChanged: updateValue(value)
                    onFocusChanged: updateValue(value)
                    onEditableChanged: updateValue(value)
                    //onActiveFocusChanged: if (!activeFocus) updateValue(value)
                }

                function updateValue(val) {
                    var index = imgControlModel.index(model.index, 0);
                    imgControlModel.setData(index, val, valueRole);
                }

            }
        }

    }
}
