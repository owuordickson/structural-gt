import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    property int txtWidthSize: 170
    property int lblWidthSize: 100

    /*model: [
        { id: "txtScalebar", text: "", labelId: "lblScalebar", labelText: "Scalebar (nm)" },
        { id: "txtPixelCount", text: "", labelId: "lblPixelCount", labelText: "Scalebar Pixel Count" },
        { id: "txtResistivity", text: "", labelId: "lblResistivity", labelText: "Resistivity (<html>&Omega;</html>m)" }
    ]*/

    model: microscopyPropsModel
    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        Label {
            id: label
            wrapMode: Text.Wrap
            Layout.preferredWidth: lblWidthSize
            text: model.text
        }

        TextField {
            id: txtField
            objectName: model.id
            Layout.fillWidth: true
            Layout.minimumWidth: txtWidthSize
            Layout.rightMargin: 10
            text: mainController.load_img_setting_val(model.id)
        }
    }
}
