import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Dialog {
    id: dialogImgFilters
    //parent: mainWindow
    anchors.centerIn: parent
    title: "Adjust Binary Filters"
    modal: true
    width: 320
    height: 350

    ColumnLayout {
        anchors.fill: parent
        ImageFilterWidget {
        }

        RowLayout {
            spacing: 10
            //Layout.topMargin: 10
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

            Button {
                Layout.preferredWidth: 54
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogImgFilters.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: "#bc0000"

                    Label {
                        text: "Cancel"
                        color: "#ffffff"
                        anchors.centerIn: parent
                    }
                }
            }

            Button {
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogImgFilters.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: "#22bc55"

                    Label {
                        text: "OK"
                        color: "#ffffff"
                        anchors.centerIn: parent
                    }
                }
            }
        }
    }
}