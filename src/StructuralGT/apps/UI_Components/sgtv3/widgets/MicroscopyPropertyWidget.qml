import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    model: [
        { id: "txtScalebar", text: "", labelId: "lblScalebar", labelText: "Scalebar (nm)" },
        { id: "txtPixelCount", text: "", labelId: "lblPixelCount", labelText: "Scalebar Pixel Count" },
        { id: "txtResistivity", text: "", labelId: "lblResistivity", labelText: "Resistivity (<html>&Omega;</html>m)" }
    ]

    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        Label {
            id: label
            wrapMode: Text.Wrap
            Layout.preferredWidth: 100
            text: modelData.labelText
        }

        TextField {
            id: txtField
            Layout.fillWidth: true
            Layout.minimumWidth: 170
            Layout.rightMargin: 10
            text: modelData.text
        }
    }
}
