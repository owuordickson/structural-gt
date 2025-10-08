import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
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
