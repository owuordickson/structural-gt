import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "components"

Rectangle {
    width: parent.width
    height: parent.height
    color: "#f0f0f0"
    border.color: "#c0c0c0"

    ColumnLayout {
        anchors.fill: parent

        TabBar {
            id: tabBar
            Layout.fillWidth: true
            //currentIndex: stackLayout.currentIndex
            TabButton { text: "Project" }
            TabButton { text: "Properties" }
            TabButton { text: "Filters" }
        }

        StackLayout {
            id: stackLayout
            currentIndex: tabBar.currentIndex

            ProjectNav{}

            ImageProperties{}

            ImageFilters{}

        }
    }

}
