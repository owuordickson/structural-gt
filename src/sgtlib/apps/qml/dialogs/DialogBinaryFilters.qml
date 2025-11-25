import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import Theme 1.0
import "../widgets"

Dialog {
    id: dialogBinFilters
    //parent: mainWindow
    anchors.centerIn: parent
    title: "Adjust Binary Filters"
    modal: true
    width: 320
    height: 210
    background: Rectangle {color: Theme.background}

    ColumnLayout {
        anchors.fill: parent
        BinaryFilterWidget {
        }


        RowLayout {
            spacing: 10
            //Layout.topMargin: 10
            Layout.alignment: Qt.AlignHCenter | Qt.AlignBottom

            Button {
                Layout.preferredWidth: 54
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogBinFilters.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.red

                    Label {
                        text: "Cancel"
                        color: Theme.white
                        anchors.centerIn: parent
                    }
                }
            }

            Button {
                Layout.preferredWidth: 40
                Layout.preferredHeight: 30
                text: ""
                onClicked: dialogBinFilters.close()

                Rectangle {
                    anchors.fill: parent
                    radius: 5
                    color: Theme.green

                    Label {
                        text: "OK"
                        color: Theme.white
                        anchors.centerIn: parent
                    }
                }
            }
        }
    }
}