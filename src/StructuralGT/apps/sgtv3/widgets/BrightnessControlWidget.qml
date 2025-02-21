import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


Item {
    id: brightnessControl  // used for external access
    height: parent.height
    width: parent.width

    property int spbWidthSize: 170
    property int lblWidthSize: 100

    // Function to get values from SpinBoxes and send to MainController
    function applyChanges() {
        let brightness_val = 0;
        let contrast_val = 0;

        // Iterate over children inside the ColumnLayout
        for (let i = 0; i < brightnessCtrlLayout.children.length; i++) {
            let elderChild = brightnessCtrlLayout.children[i];

            if (elderChild.objectName === "ctrlRowLayout") {

                // Loop through each child of the RowLayout
                for (let j = 0; j < elderChild.children.length; j++) {
                    let child = elderChild.children[j];

                    if (child.objectName === "spbBrightness") {
                        brightness_val = child.value;
                    } else if (child.objectName === "spbContrast") {
                        contrast_val = child.value;
                    }
                }
            }
        }
        //console.log("Brightness:", brightness_val, "Contrast:", contrast_val);
        mainController.adjust_brightness_contrast(brightness_val, contrast_val)
    }


    ColumnLayout {
        id: brightnessCtrlLayout
        spacing: 10

        Repeater {
            id: brightnessCtrlRepeater
            model: imgControlModel
            delegate: RowLayout {
                objectName: "ctrlRowLayout"
                Layout.fillWidth: true
                Layout.leftMargin: 10
                Layout.alignment: Qt.AlignLeft

                Label {
                    id: label
                    Layout.preferredWidth: lblWidthSize
                    text: model.text
                }

                SpinBox {
                    id: spinBox
                    objectName: model.id
                    Layout.minimumWidth: spbWidthSize
                    Layout.fillWidth: true
                    editable: true
                    from: -100
                    to: 100
                    stepSize: 1
                    value: mainController.load_img_setting_val(model.id)
                }

            }

        }
    }
}
