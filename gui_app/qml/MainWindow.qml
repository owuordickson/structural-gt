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
            Layout.alignment: Qt.AlignVCenter| Qt.AlignHCenter

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
                        var response = mainController.process_name(txtName.text);
                        lblProgress.text = response;
                        console.log(response);
                    }
                }

            }

        }

        // First row, first column (spanning 2 columns)
        Rectangle {
            Layout.row: 1
            Layout.column: 0
            Layout.columnSpan: 2
            Layout.alignment: Qt.AlignTop | Qt.AlignHCenter

            ColumnLayout {
                id: progressLayout
                spacing: 10

                Label {
                    id: lblProgress
                    Layout.preferredWidth: 100
                    text: "v1.0.0"
                }

            }

        }

    }

}