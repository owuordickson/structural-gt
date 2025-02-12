import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    model: [
        { id: "spbBrightness", labelId: "lblBrightness", labelText: "Brightness"},
        { id: "spbContrast", labelId: "lblContrast", labelText: "Contrast"}
    ]

    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        Label {
            id: label
            Layout.preferredWidth: 100
            text: modelData.labelText
        }

        SpinBox {
            id: spinBox
            Layout.minimumWidth: 170
            Layout.fillWidth: true
            from: 0
            to: 100
            stepSize: 1
            value: 0
        }

    }

}
