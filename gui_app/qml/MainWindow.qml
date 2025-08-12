import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    id: mainWindow
    width: 1024
    height: 800
    visible: true
    title: "GUI Tutorial"

    /*Component.onCompleted: {
        console.log("Checking controller:", mainController);
        if (!mainController) {
            console.error("mainController is undefined!");
        }
        mainController.process_name("Dickson Owuor");
    }*/

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
            width: 956
            height: 384
            color: "transparent"

            ColumnLayout {
                id: loginControlLayout
                spacing: 10
                anchors.centerIn: parent
                visible: true

                Label {
                    id: lblName
                    Layout.preferredWidth: 100
                    text: "Add Image"
                }

                /*TextField {
                    id: txtName
                    Layout.preferredWidth: 100
                    text: ""
                }*/

                Button {
                    id: btnOK
                    text: "OK"
                    onClicked: {
                        /*lblName.text = "Welcome " + txtName.text;
                        var response = mainController.process_name(txtName.text);
                        lblProgress.text = response;
                        console.log(response);*/
                        mainController.process_image();
                    }
                }

            }

            ColumnLayout {
                id: filterControlLayout
                spacing: 10
                anchors.centerIn: parent
                visible: false

                ImageFilterWidget {}
            }

        }

        // First row, first column (spanning 2 columns)
        Rectangle {
            Layout.row: 1
            Layout.column: 0
            Layout.columnSpan: 2
            Layout.alignment: Qt.AlignVCenter| Qt.AlignHCenter
            width: 956
            height: 384
            color: "transparent"

            ColumnLayout {
                id: progressLayout
                anchors.centerIn: parent
                spacing: 10

                Label {
                    id: lblError
                    text: "No image to display!"
                    color: "#FF0000"
                    visible: true
                }

                Rectangle {
                    id: imgContainer
                    width: 256
                    height: 256
                    color: "lightgray"
                    visible: false

                    Image {
                        id: imgView
                        anchors.centerIn: parent
                        source: ""
                        // Scale to fit while keeping aspect ratio
                        fillMode: Image.PreserveAspectFit
                        // Prevent overflow
                        width: parent.width
                        height: parent.height
                        clip: true
                    }
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
            lblProgress.text = val + "%: " + msg;
            console.log(val + "%: " + msg);
       }

       function onImageChangedSignal(show) {
            if (show) {
                lblError.visible = false;
                imgContainer.visible = true;
                imgView.source = mainController.get_pixmap();
                loginControlLayout.visible = false;
                filterControlLayout.visible = true;
            } else {
                lblError.visible = true;
                imgContainer.visible = false;
                loginControlLayout.visible = true;
                filterControlLayout.visible = false;
            }
       }

    }

}