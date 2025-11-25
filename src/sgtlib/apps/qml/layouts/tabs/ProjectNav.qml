import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0
import "../../widgets"

Rectangle {
    color: Theme.background
    border.color: Theme.borderColor
    Layout.fillWidth: true
    Layout.fillHeight: true

    ColumnLayout {
        id: colImgProjNavLayout
        anchors.fill: parent

        ProjectFoldersWidget {
        }

        Rectangle {
            height: 1
            color: "#d0d0d0"
            Layout.fillWidth: true
            Layout.alignment: Qt.AlignHCenter
            Layout.topMargin: 5
            Layout.leftMargin: 20
            Layout.rightMargin: 20
        }

        ImageThumbnailWidget {
        }
    }

}
