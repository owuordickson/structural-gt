import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: mainWindow
    width: 1024
    height: 800
    visible: true
    title: "GUI Tutorial"

    Component.onCompleted: {
        console.log("Checking controller:", mainController);
        if (!mainController) {
            console.error("mainController is undefined!");
        }
        //mainController.process_name("Dickson Owuor");
    }

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
                        //mainController.process_image();
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
                    id: lblError
                    text: "No image to display!"
                    color: "#FF0000"
                    visible: true
                }

                Image {
                    id: imgView
                    width: 512
                    height: 512
                    fillMode: Image.PreserveAspectFit
                    source: ""
                    visible: false
                }

                Label {
                    id: lblProgress
                    Layout.preferredWidth: 100
                    text: "v1.0.0"
                }

            }

        }

    }

    Connections {
        target: mainController

       function onUpdateProgress(val, msg) {
            lblProgress.text = (val + "%: " + msg);
            console.log(val + "%: " + msg);
       }

       function onImageChangedSignal(show) {
            if (show) {
                lblError.visible = false;
                imgView.visible = true;
                imgView.source = mainController.get_pixmap();
            } else {
                lblError.visible = true;
                imgView.visible = false;
            }
       }

    }

}