import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Rectangle {
    color: "#f0f0f0"
    border.color: "#c0c0c0"
    Layout.fillWidth: true
    Layout.fillHeight: true

    ScrollView {
        //width: parent.width
        height: parent.height

        ColumnLayout {
            //anchors.fill: parent

            Text {
                text: "Image Properties"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }

            ImagePropertyWidget{}

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
                text: "Graph Properties"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }

            GraphPropertyWidget{}


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
                text: "Microscopy Properties"
                font.pixelSize: 12
                font.bold: true
                Layout.topMargin: 10
                Layout.bottomMargin: 5
                Layout.alignment: Qt.AlignHCenter
            }

            MicroscopyPropertyWidget{}
        }
    }

}
