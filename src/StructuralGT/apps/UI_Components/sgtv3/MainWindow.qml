import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "widgets"

ApplicationWindow {
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
            Layout.preferredWidth: parent.width * 0.3
            Layout.fillWidth: true
            Layout.fillHeight: true
            LeftContent{}
        }

        // Second row, second column
        Rectangle {
            Layout.row: 1
            Layout.column: 1
            Layout.rightMargin: 10
            Layout.preferredHeight: parent.height-40
            Layout.preferredWidth: parent.width * 0.7
            Layout.fillWidth: true
            Layout.fillHeight: true
            RightContent{}
        }
    }

    Dialog {
        id: aboutDialog
        title: "About This App"
        modal: true
        standardButtons: Dialog.Ok

        Label {
            text: "StructuralGT v3.0.1\nCopyright (C) 2024\nthe Regents of the University of Michigan."
            anchors.centerIn: parent
        }
    }

}
