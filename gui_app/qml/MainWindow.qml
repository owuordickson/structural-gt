import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: mainWindow
    width: 1024
    height: 800
    visible: true
    title: "GUI Tutorial"

    GridLayout {
        anchors.fill: parent
        rows: 2
        columns: 2

        // First row, first column (spanning 2 columns) 
        Rectangle {
            Layout.row: 0
            Layout.column: 0
            Layout.columnSpan: 2
            //Layout.leftMargin: 10
            //Layout.rightMargin: 10
            Layout.alignment: Qt.AlignTop | Qt.AlignHCenter
            //Layout.preferredHeight: 100
            //Layout.preferredWidth: parent.width
            //Layout.fillWidth: true
            //Layout.fillHeight: true

            ColumnLayout {
                id: loginControlLayout
                spacing: 10

                Label {
                    id: lblName
                    Layout.preferredWidth: 100
                    text: "What is your name?"
                }

                TextField {
                    id: txtName
                    Layout.preferredWidth: 100
                    text: ""
                }

                Button {
                    id: btnOK
                    text: "OK"
                    onClicked: {
                        lblName.text = "Welcome " + txtName.text;
                        mainController.process_name(txtName.text);
                    }
                }

            }

        }

    }

}