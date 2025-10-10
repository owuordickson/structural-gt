import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    Layout.fillWidth: true
    Layout.fillHeight: true

    property int lblWidthSize: 280

    ColumnLayout {
        id: colImgFiltersLayout
        anchors.fill: parent

        AIModeWidget {
        }


        BinaryFilterWidget {
        }

        Rectangle {
            id: rectHLine1
            height: 1
            color: "#d0d0d0"
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            Layout.leftMargin: 20
            Layout.rightMargin: 20
            visible: imageController.display_image()
        }
        ImageFilterWidget {
        }

        FiltersWidget {
        }

    }

    Connections {
        target: mainController

        function onImageChangedSignal() {
            // Force refresh
            rectHLine1.visible = imageController.display_image();
        }
    }
}
