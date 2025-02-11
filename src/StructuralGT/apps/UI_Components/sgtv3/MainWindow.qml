import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

ApplicationWindow {
    width: 1024
    height: 800
    visible: true
    title: "Structural GT"

    menuBar: MenuBarComponent{}

    footer: StatusBarComponent{}

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
            RibbonComponent {}
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

}
