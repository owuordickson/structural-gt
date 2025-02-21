import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    model: gtcListModel
    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        CheckBox {
            id: checkBox
            objectName: model.id
            //Layout.preferredWidth: 100
            text: model.text
            checked: mainController.get_selected_gtc_val(model.id)
        }
    }

}
