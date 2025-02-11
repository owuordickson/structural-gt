import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Rectangle {
    width: parent.width
    height: parent.height
    color: "#f0f0f0"
    border.color: "#d0d0d0"

    ColumnLayout {
        anchors.fill: parent

        TabBar {
            id: tabBar
            Layout.fillWidth: true
            //currentIndex: stackLayout.currentIndex
            TabButton { text: "Project" }
            TabButton { text: "Image" }
            TabButton { text: "Filters" }
        }

        StackLayout {
            id: stackLayout
            currentIndex: tabBar.currentIndex

            Rectangle {
                color: "lightgray"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Text {
                    text: "Content of Tab 1"
                    anchors.centerIn: parent
                }
            }

            Rectangle {
                color: "lightblue"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Text {
                    text: "Content of Tab 2"
                    anchors.centerIn: parent
                }
            }

            Rectangle {
                color: "lightgreen"
                Layout.fillWidth: true
                Layout.fillHeight: true
                Text {
                    text: "Content of Tab 3"
                    anchors.centerIn: parent
                }
            }
        }
    }

}
