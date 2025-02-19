import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Repeater {

    property int cbxWidthSize: 100
    property int spbWidthSize: 170
    property int sldWidthSize: 140
    property int lblWidthSize: 50

    model: [
        { id: "cbxAutolevel", text: "Autolevel", dataId: "sldAutolevel", labelId: "lblAutolevel" },
        { id: "cbxGaussian", text: "Gaussian", dataId: "sldGaussian", labelId: "lblGaussian" },
        { id: "cbxLaplacian", text: "Laplacian", dataId: "sldLaplacian", labelId: "lblLaplacian" },
        { id: "cbxLowpass", text: "Lowpass", dataId: "sldLowpass", labelId: "lblLowpass" },
        { id: "cbxGamma", text: "LUT Gamma", dataId: "sldGamma", labelId: "lblGamma" },
        { id: "cbxMedian", text: "Median", dataId: "sldMedian", labelId: "lblMedian" },
        { id: "cbxScharr", text: "Scharr", dataId: "sldScharr", labelId: "lblScharr" },
        { id: "cbxSobel", text: "Sobel", dataId: "sldSobel", labelId: "lblSobel" }
    ]

    delegate: RowLayout {
        Layout.fillWidth: true
        Layout.leftMargin: 10
        Layout.alignment: Qt.AlignLeft

        CheckBox {
            id: checkBox
            objectName: modelData.id
            Layout.preferredWidth: cbxWidthSize
            text: modelData.text
            checked: modelData.id === "cbxGamma"
        }

        Loader {
            id: controlLoader
            sourceComponent: modelData.id === "cbxLowpass" ? spinComponent : sliderComponent
        }

        Component {
            id: spinComponent

            RowLayout {
                Layout.fillWidth: true
                SpinBox {
                    id: spinbox
                    objectName: modelData.dataId
                    Layout.minimumWidth: spbWidthSize
                    Layout.fillWidth: true
                    from: 0
                    to: 100
                    stepSize: 1
                    value: 11
                    enabled: checkBox.checked
                }
            }
        }

        Component {
            id: sliderComponent

            RowLayout {
                Layout.fillWidth: true

                Slider {
                    id: slider
                    objectName: modelData.dataId
                    Layout.minimumWidth: sldWidthSize
                    Layout.fillWidth: true
                    from: 0
                    to: 5.0
                    stepSize: 0.01
                    value: 1.0
                    enabled: checkBox.checked
                }

                Label {
                    id: label
                    Layout.preferredWidth: lblWidthSize
                    text: Number(slider.value).toFixed(2) // Display one decimal place
                    enabled: checkBox.checked
                }
            }
        }
    }
}
