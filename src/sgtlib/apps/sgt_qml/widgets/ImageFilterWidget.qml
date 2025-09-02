import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: imgFiltersControl
    spacing: 10
    Layout.preferredHeight: 250
    Layout.preferredWidth: parent.width
    visible: mainController.display_image()
    enabled: mainController.enable_img_controls()

    property int cbxWidthSize: 100
    property int spbWidthSize: 170
    property int sldWidthSize: 140
    property int lblWidthSize: 50
    property int valueRole: Qt.UserRole + 4
    property int dataValueRole: Qt.UserRole + 6

    Repeater {
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
                property bool isChecked: model.value
                property bool clickedChange: false  // Flag
                checked: isChecked
                onCheckedChanged: {
                    if (clickedChange) {
                        // Reset flag so the next programmatic change will run normally
                        clickedChange = false
                        return
                    }

                    if (isChecked !== checked) {  // Only update if there is a change
                        isChecked = checked
                        let val = checked ? 1 : 0;
                        var index = imgFilterModel.index(model.index, 0);
                        imgFilterModel.setData(index, val, valueRole);
                        mainController.apply_changes(""); // Only runs if not from click
                    }
                }
                onClicked: {
                    clickedChange = true;
                    isChecked = checked
                    let val = checked ? 1 : 0;
                    var index = imgFilterModel.index(model.index, 0);
                    imgFilterModel.setData(index, val, valueRole);
                    mainController.apply_changes("binary");
                }
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
                        property var currSBVal: model.dataValue
                        value: currSBVal
                        onValueChanged: updateValue(currSBVal, value)
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
                        property var currVal: model.dataValue
                        value: currVal
                        onValueChanged: updateValue(currVal, value)
                    }

                    Label {
                        id: label
                        Layout.preferredWidth: lblWidthSize
                        text: model.stepSize >= 1 ? Number(slider.value).toFixed(0) : Number(slider.value).toFixed(2) // Display 2 decimal place
                        enabled: checkBox.checked
                    }
                }
            }

            function updateValue(curr_val, val) {
                if (curr_val !== val) {
                    curr_val = val;
                    var index = imgFilterModel.index(model.index, 0);
                    imgFilterModel.setData(index, val, dataValueRole);
                    mainController.apply_changes("");
                }
            }
        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgFiltersControl.visible = mainController.display_image();
            imgFiltersControl.enabled = mainController.enable_img_controls();
        }

    }

}
