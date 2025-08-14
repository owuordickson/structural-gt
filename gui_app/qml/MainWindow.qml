import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "layouts"

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
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            width: 956
            height: 384
            color: "transparent"

            TopLayout {}
        }

        // Second row, first column (spanning 2 columns)
        Rectangle {
            Layout.row: 1
            Layout.column: 0
            Layout.columnSpan: 2
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            width: 956
            height: 384
            color: "transparent"

            BottomLayout {}
        }
    }
}
