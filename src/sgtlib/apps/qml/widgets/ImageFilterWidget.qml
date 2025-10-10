import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ColumnLayout {
    id: imgFiltersControl
    Layout.leftMargin: 10
    Layout.preferredHeight: 250
    Layout.preferredWidth: parent.width
    Layout.alignment: Qt.AlignTop

    property int cbxWidthSize: 100
    property int spbWidthSize: 170
    property int sldWidthSize: 140
    property int lblWidthSize: 50
    property int valueRole: Qt.UserRole + 4
    property int dataValueRole: Qt.UserRole + 6

    Label {
        id: lblNoImgFilters
        Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter
        text: "No image filters to show!\nCreate project/add image."
        color: "#808080"
        visible: !imageController.display_image()
    }

    Text {
        id: txtTitleImgFilters
        text: "Image Filters"
        font.pixelSize: 12
        font.bold: true
        Layout.alignment: Qt.AlignHCenter
        visible: imageController.display_image()
    }

    ColumnLayout {
        id: colFilters
        spacing: 10
        visible: imageController.display_image()

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
                            imageController.apply_changes(""); // Only runs if not from click
                        }
                    }
                    onClicked: {
                        clickedChange = true;
                        isChecked = checked
                        let val = checked ? 1 : 0;
                        var index = imgFilterModel.index(model.index, 0);
                        imgFilterModel.setData(index, val, valueRole);
                        imageController.apply_changes("binary");
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
                        imageController.apply_changes("");
                    }
                }
            }

        }
    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            lblNoImgFilters.visible = !imageController.display_image();
            txtTitleImgFilters.visible = imageController.display_image();
            colFilters.visible = imageController.display_image();
            colFilters.enabled = imageController.enable_img_controls();
        }

    }

}
