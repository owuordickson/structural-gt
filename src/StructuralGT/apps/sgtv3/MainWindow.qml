import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
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
            mainController.apply_img_ctrl_changes()
            dialogBrightnessCtrl.close()
        }

        /*onRejected: {
            dialogController.reject()
            dialogBrightnessCtrl.close()
        }*/
    }


    Dialog {
        id: dialogShowGraph
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

        /*onAccepted: {
            dialogController.accept()  // In Python side
            dialog.close()
        }

        onRejected: {
            dialogController.reject()
            dialog.close()
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

        /*onAccepted: {
            dialogController.accept()  // In Python side
            dialog.close()
        }

        onRejected: {
            dialogController.reject()
            dialog.close()
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

        /*onAccepted: {
            dialogController.accept()  // In Python side
            dialog.close()
        }

        onRejected: {
            dialogController.reject()
            dialog.close()
        }*/
    }

    Dialog {
        id: dialogGTOptions
        //parent: mainWindow
        anchors.centerIn: parent
        title: "Select Graph Computations"
        modal: true
        standardButtons: Dialog.Ok | Dialog.Cancel
        width: 240
        height: 400

        ColumnLayout {
            anchors.fill: parent

            GTWidget{}
        }

        /*onAccepted: {
            dialogController.accept()  // In Python side
            dialog.close()
        }

        onRejected: {
            dialogController.reject()
            dialog.close()
        }*/
    }

}


//about = A software tool that allows graph theory analysis of nano-structures. This is a modified version of StructuralGT initially proposed by Drew A. Vecchio, DOI: 10.1021/acsnano.1c04711.
