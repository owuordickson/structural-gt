import QtQuick
import QtQuick.Controls 2.15
import QtQuick.Controls.Material 2.15 as MaterialControls
import QtQuick.Layouts
//import QtQuick.Controls.Basic as Basic
import "components"

Rectangle {
    width: 300
    height: parent.height
    color: "#f0f0f0"
    border.color: "#c0c0c0"

    ColumnLayout {
        anchors.fill: parent

        MaterialControls.TabBar {
            id: tabBar
            currentIndex: 2
            Layout.fillWidth: true
            TabButton { text: "Project" }
            TabButton { text: "Properties" }
            TabButton { text: "Filters" }
        }

        StackLayout {
            id: stackLayout
            //width: parent.width
            Layout.fillWidth: true
            currentIndex: tabBar.currentIndex


            ProjectNav{}

            ImageProperties{}

            ImageFilters{}


        }
    }

    Connections {
        target: mainController

        function onProjectOpenedSignal(name) {
            tabBar.currentIndex = 0;
        }
    }
}
