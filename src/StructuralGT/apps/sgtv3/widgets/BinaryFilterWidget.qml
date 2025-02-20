import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    Layout.fillWidth: true
    Layout.leftMargin: 10
    Layout.alignment: Qt.AlignLeft

    property int btnWidthSize: 100
    property int spbWidthSize: 170
    property int sldWidthSize: 140
    property int lblWidthSize: 50
    property int checkedBtnId: mainController.load_img_setting_val("threshold_type")

    ButtonGroup {
        id: btnGroup
        checkedButton: checkedBtnId === 0 ? rdoGlobal : checkedBtnId === 1 ? rdoAdaptive : checkedBtnId === 2 ? rdoOtsu : rdoGlobal
    }


    RowLayout {

        RadioButton {
            id:rdoAdaptive
            text: "Adaptive"
            ButtonGroup.group: btnGroup
            Layout.preferredWidth: btnWidthSize
        }

        SpinBox {
            // ONLY ODD NUMBERS
            id: spbAdaptive
            Layout.minimumWidth: spbWidthSize
            //Layout.fillWidth: true
            from: 1
            to: 999
            stepSize: 2
            value: mainController.load_img_setting_val("adaptive_local_threshold_value")
            enabled: rdoAdaptive.checked
            editable: true  // Allow user input
            onValueChanged: {
                if (value % 2 === 0) {
                    value = value - 1;  // Convert even input to nearest odd
                }
            }
            validator: IntValidator { bottom: spbAdaptive.from; top: spbAdaptive.to }
        }
    }

    RowLayout {

        RadioButton {
            id: rdoGlobal
            text: "Global"
            ButtonGroup.group: btnGroup
            Layout.preferredWidth: 100
        }

        Slider {
            id: sldGlobal
            Layout.minimumWidth: sldWidthSize
            Layout.fillWidth: true
            from: 1
            to: 255
            stepSize: 1
            value: mainController.load_img_setting_val("global_threshold_value")
            enabled: rdoGlobal.checked
        }

        Label {
            id: lblGlobal
            Layout.preferredWidth: lblWidthSize
            text: Number(sldGlobal.value).toFixed(0) // Display one decimal place
            enabled: rdoGlobal.checked
        }

    }

    RadioButton {
        id: rdoOtsu
        text: "OTSU"
        ButtonGroup.group: btnGroup
        Layout.preferredWidth: btnWidthSize
    }

    CheckBox {
        id: cbxDarkFg
        text: "Apply Dark Foreground"
        checked: mainController.load_img_setting("apply_dark_foreground")
    }
}
