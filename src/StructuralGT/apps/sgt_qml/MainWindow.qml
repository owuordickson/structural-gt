import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import QtQuick.Dialogs as QuickDialogs
import Qt.labs.platform as Platform
import "widgets"

ApplicationWindow {
    id: mainWindow
    width: 1024
    height: 800
    visible: true
    title: "Structural GT"

    menuBar: MenuBarWidget{}

    footer: StatusBarWidget{}

    GridLayout {
        anchors.fill: parent
        rows: 2
        columns: 2

        // First row, first column (spanning 2 columns)
        Rectangle {
            Layout.row: 0
            Layout.column: 0
            Layout.columnSpan: 2
            Layout.leftMargin: 10
            Layout.rightMargin: 10
            Layout.alignment: Qt.AlignTop
            Layout.preferredHeight: 40
            Layout.preferredWidth: parent.width
            Layout.fillWidth: true
            Layout.fillHeight: true
            RibbonWidget {}
        }

        // Second row, first column
        Rectangle {
            id: recLeftPane
            Layout.row: 1
            Layout.column: 0
            Layout.leftMargin: 10
            Layout.rightMargin: 5
            Layout.preferredHeight: parent.height - 40
            //Layout.preferredWidth: parent.width * 0.3
            Layout.preferredWidth: 300
            Layout.fillWidth: true
            Layout.fillHeight: true
            LeftContent{}
        }

        // Second row, second column
        Rectangle {
            id: recRightPane
            Layout.row: 1
            Layout.column: 1
            Layout.rightMargin: 10
            Layout.preferredHeight: parent.height - 40
            //Layout.preferredWidth: parent.width * 0.7
            Layout.preferredWidth: parent.width - 300
            Layout.fillWidth: true
            Layout.fillHeight: true
            RightContent{}
        }
    }

    function toggleLeftPane (showVal) {
        recLeftPane.visible = showVal;
    }


    Dialog {
        id: dialogAbout
        //parent: mainWindow
        title: "About this software"
        modal: true
        standardButtons: Dialog.Ok
        anchors.centerIn: parent
        width: 300
        height: 200

        Label {
            text: "StructuralGT v3.0.1\nCopyright (C) 2024\nthe Regents of the University of Michigan."
            anchors.centerIn: parent
        }
    }

    Dialog {
        id: dialogAlert
        //parent: mainWindow
        title: ""
        modal: true
        standardButtons: Dialog.Ok
        anchors.centerIn: parent
        width: 300
        height: 200

        /*contentItem: ColumnLayout {
                spacing: 10
                width: parent.width

                // Custom Header for the Dialog Title
                Label {
                    text: dialogAlert.title
                    font.bold: true
                    font.pointSize: 14
                    color: "red"  // Change title color to red
                    horizontalAlignment: Text.AlignHCenter
                    Layout.fillWidth: true  // Ensure it spans the full width
                }

                Label {
                    id: lblAlertMsg
                    width: parent.width
                    wrapMode: Text.Wrap  // Enable text wrapping
                    anchors.centerIn: parent
                    leftPadding: 10
                    rightPadding: 10
                    horizontalAlignment: Text.AlignJustify  // Justify the text
                    color: "#bc2222"
                    text: ""
                }
            }*/

        Label {
            id: lblAlertMsg
            width: parent.width
            wrapMode: Text.Wrap  // Enable text wrapping
            anchors.centerIn: parent
            leftPadding: 10
            rightPadding: 10
            horizontalAlignment: Text.AlignJustify  // Justify the text
            color: "#bc2222"
            text: ""
        }
    }

    Dialog {
        id: createProjectDialog
        anchors.centerIn: parent
        title: "Create SGT Project"
        modal: true
        width: 300
        height: 150

        ColumnLayout {
            anchors.fill: parent
            CreateProjectWidget { id: createProjectControls }

            RowLayout {
                spacing: 10
                //Layout.topMargin: 10
                Layout.alignment: Qt.AlignHCenter
                Button {
                    text: "OK"
                    onClicked: {
                        var name = createProjectControls.txtName.text;
                        var location = createProjectControls.txtLocation.text;

                        if (name === "") {
                            //console.log("Please fill in all fields.");
                            createProjectControls.lblName.text = "Name*";
                            createProjectControls.lblName.color = "red";
                            createProjectControls.txtName.placeholderText = "please enter a name!"

                        } else if (location === "") {
                            createProjectControls.lblLocation.text = "Location*";
                            createProjectControls.lblLocation.color = "red";

                        } else {
                            mainController.create_sgt_project(name, location);
                            createProjectDialog.close();
                        }
                    }
                }
                Button {
                    text: "Cancel"
                    onClicked: createProjectDialog.close()
                }
            }
        }
    }

    /*Dialog {
        id: saveProjectDialog
        anchors.centerIn: parent
        title: "Save SGT Project"
        modal: true
        width: 300
        height: 150

        ColumnLayout {
            anchors.fill: parent
            CreateProjectWidget { id: saveProjectControls }

            RowLayout {
                spacing: 10
                //Layout.topMargin: 10
                Layout.alignment: Qt.AlignHCenter
                Button {
                    text: "OK"
                    onClicked: {
                        var name = saveProjectControls.txtName.text;
                        var location = saveProjectControls.txtLocation.text;

                        if (name === "") {
                            saveProjectControls.lblName.text = "Name*";
                            saveProjectControls.lblName.color = "red";
                            saveProjectControls.txtName.placeholderText = "please enter a name!"

                        } else if (location === "") {
                            saveProjectControls.lblLocation.text = "Location*";
                            saveProjectControls.lblLocation.color = "red";

                        } else {
                            mainController.create_sgt_project(name, location);
                            saveProjectDialog.close();
                        }
                    }
                }
                Button {
                    text: "Cancel"
                    onClicked: saveProjectDialog.close()
                }
            }
        }
    }*/

    Dialog {
        id: dialogBrightnessCtrl
        //parent: mainWindow
        anchors.centerIn: parent
        title: "Control Brightness/Contrast"
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 300
        height: 150

        ColumnLayout {
            anchors.fill: parent
            BrightnessControlWidget{ id: brightnessControl }
        }

        onAccepted: {
            //mainController.apply_img_ctrl_changes();
            dialogBrightnessCtrl.close()
        }

        /*onRejected: {
            dialogController.reject()
            dialogBrightnessCtrl.close()
        }*/
    }

    Dialog {
        id: dialogExtractGraph
        //parent: mainWindow
        anchors.centerIn: parent
        title: "Graph Extraction Options"
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 300
        height: 400

        ScrollView {
            width: parent.width
            height: parent.height

            GraphExtractWidget{}
        }

        onAccepted: {
            mainController.run_extract_graph();
            dialogExtractGraph.close()
        }

        /*onRejected: {
            dialogExtractGraph.close()
        }*/
    }

    Dialog {
        id: dialogBinFilters
        //parent: mainWindow
        anchors.centerIn: parent
        title: "Adjust Binary Filters"
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 300
        height: 200

        ColumnLayout {
            anchors.fill: parent
            BinaryFilterWidget{}
        }

        onAccepted: {
            dialogBinFilters.close()
        }

        /*onRejected: {
            dialogBinFilters.close()
        }*/
    }

    Dialog {
        id: dialogImgFilters
        //parent: mainWindow
        anchors.centerIn: parent
        title: "Adjust Binary Filters"
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 300
        height: 400

        ColumnLayout {
            anchors.fill: parent
            ImageFilterWidget{}
        }

        onAccepted: {
            dialogImgFilters.close()
        }

        /*onRejected: {
            dialogImgFilters.close()
        }*/
    }

    Dialog {
        id: dialogRunAnalyzer
        //parent: mainWindow
        anchors.centerIn: parent
        title: "Select Graph Computations"
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 240
        height: 500

        ColumnLayout {
            anchors.fill: parent
            GTWidget{}
        }

        onAccepted: {
            mainController.run_graph_analyzer();
            dialogRunAnalyzer.close()
        }

        /*onRejected: {
            dialogRunAnalyzer.close()
        }*/
    }

    Dialog {
        id: dialogRunMultiAnalyzer
        //parent: mainWindow
        anchors.centerIn: parent
        title: "Select Graph Computations"
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 240
        height: 500

        ColumnLayout {
            anchors.fill: parent
            GTWidget{}
        }

        onAccepted: {
            mainController.run_multi_graph_analyzer();
            dialogRunAnalyzer.close()
        }

        /*onRejected: {
            dialogRunAnalyzer.close()
        }*/
    }

    Platform.FolderDialog {
        id: outFolderDialog
        title: "Select a Folder"
        onAccepted: {
            //console.log("Selected folder:", folder)
            mainController.set_output_dir(folder)
        }
        //onRejected: {console.log("Canceled")}
    }

    Platform.FolderDialog {
        id: imageFolderDialog
        title: "Select a Folder"
        onAccepted: {
            //console.log("Selected folder:", folder)
            mainController.add_multiple_images(imageFolderDialog.folder);
        }
        //onRejected: {console.log("Canceled")}
    }

    QuickDialogs.FileDialog {
        id: imageFileDialog
        title: "Open file"
        nameFilters: ["Image files (*.jpg *.tif *.png *.jpeg)"]
        onAccepted: {
            //console.log("Selected file:", fileDialog.selectedFile)
            mainController.add_single_image(imageFileDialog.selectedFile);
        }
        //onRejected: console.log("File selection canceled")
    }

    QuickDialogs.FileDialog {
        id: projectFileDialog
        title: "Open .sgtproj file"
        nameFilters: ["Project files (*.sgtproj)"]
        onAccepted: {
            mainController.open_sgt_project(projectFileDialog.selectedFile);
        }
        //onRejected: console.log("File selection canceled")
    }


    Connections {
        target: mainController

        function onShowAlertSignal(title, msg) {
            dialogAlert.title = title;
            lblAlertMsg.text = msg;
            lblAlertMsg.color = "#2255bc";
            dialogAlert.open();
        }

    }
}


//about = A software tool that allows graph theory analysis of nano-structures. This is a modified version of StructuralGT initially proposed by Drew A. Vecchio, DOI: 10.1021/acsnano.1c04711.
