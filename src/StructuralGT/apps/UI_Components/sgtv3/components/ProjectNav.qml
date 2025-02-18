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
        width: parent.width
        height: parent.height

        ScrollView {
            //width: parent.width
            height: parent.height

            ColumnLayout {
                //anchors.fill: parent

                RowLayout {
                    Layout.topMargin: 10
                    Layout.bottomMargin: 5

                    Label {
                        text: "Output Dir:"
                        font.bold: true
                    }

                    TextField {
                        id: txtOutputDir
                        text: ""
                    }

                    /*Button {
                        id: btnChangeOutDir
                        text: "Change"
                    }*/
                }


                Button {
                    id: btnImportImages
                    Layout.alignment: Qt.AlignLeft
                    text: "Import image(s)"
                }

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
                    text: "Image List"
                    font.pixelSize: 12
                    font.bold: true
                    Layout.topMargin: 10
                    Layout.bottomMargin: 5
                    Layout.alignment: Qt.AlignHCenter
                }

                ProjectWidget {}

            }

        }

    }



}
