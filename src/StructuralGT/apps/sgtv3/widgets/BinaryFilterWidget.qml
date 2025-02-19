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

    ButtonGroup {
        id: btnGroup
    }


    RowLayout {

        RadioButton {
            id:rdoAdaptive
            text: "Adaptive"
            ButtonGroup.group: btnGroup
            Layout.preferredWidth: btnWidthSize
        }

        SpinBox {
            id: spbAdaptive
            Layout.minimumWidth: spbWidthSize
            //Layout.fillWidth: true
            from: 0
            to: 100
            stepSize: 1
            value: 11
            enabled: rdoAdaptive.checked
        }

    }

    RowLayout {

        RadioButton {
            id: rdoGlobal
            text: "Global"
            checked: true
            ButtonGroup.group: btnGroup
            Layout.preferredWidth: 100
        }

        Slider {
            id: sldGlobal
            Layout.minimumWidth: sldWidthSize
            Layout.fillWidth: true
            from: 0
            to: 255
            stepSize: 1
            value: 127
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
    }
}
