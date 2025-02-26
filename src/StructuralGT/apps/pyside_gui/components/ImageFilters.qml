import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    Layout.fillWidth: true
    Layout.fillHeight: true

    property int lblWidthSize: 280

    ScrollView {
        //width: parent.width
        height: parent.height

        GridLayout {
            //anchors.fill: parent
            columns: 1

            Text {
                text: "Binary Filters"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }

            Label {
                id: lblNoImgFilters
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 20
                text: "No image filters to show!\nCreate project/add image."
                color: "#808080"
                visible: !mainController.display_image();
            }
            BinaryFilterWidget{}

            Rectangle {
                height: 1
                color: "#d0d0d0"
                Layout.fillWidth: true
                Layout.alignment: Qt.AlignHCenter
                Layout.topMargin: 20
                Layout.leftMargin: 20
                Layout.rightMargin: 20
            }

            Text {
                text: "Image Filters"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }


            ImageFilterWidget{}

        }
    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            lblNoImgFilters.visible = !mainController.display_image();
        }

    }
}
