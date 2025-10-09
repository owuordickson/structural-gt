import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Controls.Imagine as Imagine

ColumnLayout {
    id: imgFilterButtons
    Layout.preferredHeight: 28
    Layout.preferredWidth: parent.width
    Layout.alignment: Qt.AlignTop
    Layout.bottomMargin: 10
    visible: imageController.display_image()
    enabled: imageController.enable_img_controls()


    RowLayout {
        Layout.alignment: Qt.AlignHCenter

        Imagine.Button {
            id: btnShowImgHistogram
            text: "Image Histogram"
            padding: 5
            enabled: imageController.enable_img_controls()
            onClicked: imgHistogramWindow.visible = true
        }

        Rectangle {
            width: 1
            height: 18
            color: "#d0d0d0"
        }

        Imagine.Button {
            id: btnShowImgColors
            text: "Image Colors"
            padding: 5
            enabled: imageController.enable_img_controls()
            onClicked: imgColorsWindow.visible = true
        }

    }


    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            imgFilterButtons.visible = imageController.display_image();
            imgFilterButtons.enabled = imageController.enable_img_controls();
        }
    }

    Connections {
        target: imageController

        function onShowImageFilterControls(allow) {

            if (allow) {
                btnShowImgHistogram.enabled = imageController.enable_img_controls();
                btnShowImgColors.enabled = imageController.enable_img_controls();
            } else {
                btnShowImgHistogram.enabled = allow;
                btnShowImgColors.enabled = allow;
            }

        }

    }

}