import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    property int cbxWidthSize: 100
    property int spbWidthSize: 170
    property int sldWidthSize: 140
    property int lblWidthSize: 50

    model: imgFilterModel
    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        CheckBox {
            id: checkBox
            objectName: model.id
            Layout.preferredWidth: cbxWidthSize
            text: model.text
            checked: mainController.load_img_setting(model.id)
        }

        Loader {
            id: controlLoader
            sourceComponent: (model.id === "apply_median_filter" || model.id === "apply_scharr_gradient") ? blankComponent : model.id === "apply_lowpass_filter" ? spinComponent : sliderComponent
        }

        Component {
            id: blankComponent

            RowLayout {
                Layout.fillWidth: true
                Layout.bottomMargin: 10
            }
        }

        Component {
            id: spinComponent

            RowLayout {
                Layout.fillWidth: true
                SpinBox {
                    id: spinbox
                    objectName: model.dataId
                    Layout.minimumWidth: spbWidthSize
                    Layout.fillWidth: true
                    enabled: checkBox.checked
                    from: model.minValue
                    to: model.maxValue
                    stepSize: model.stepSize
                    value: mainController.load_img_setting_val(model.id)
                }
            }
        }

        Component {
            id: sliderComponent

            RowLayout {
                Layout.fillWidth: true

                Slider {
                    id: slider
                    objectName: model.dataId
                    Layout.minimumWidth: sldWidthSize
                    Layout.fillWidth: true
                    enabled: checkBox.checked
                    from: model.minValue
                    to: model.maxValue
                    stepSize: model.stepSize
                    value: mainController.load_img_setting_val(model.id)
                }

                Label {
                    id: label
                    Layout.preferredWidth: lblWidthSize
                    text: model.stepSize >= 1 ? Number(slider.value).toFixed(0) : Number(slider.value).toFixed(2) // Display 2 decimal place
                    enabled: checkBox.checked
                }
            }
        }
    }
}
